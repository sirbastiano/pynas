import pandas as pd
import numpy as np
import pickle
from copy import deepcopy
import tqdm, os

from ..blocks.heads import MultiInputClassifier
from .individual import Individual 
from .generic_unet import GenericUNetNetwork
from ..opt.evo import single_point_crossover, gene_mutation
from .generic_lightning_module import GenericLightningSegmentationNetwork, GenericLightningNetwork
import logging 

import torch
import torch.nn as nn
import pytorch_lightning as pl

from IPython.display import clear_output


class Population:
    def __init__(self, n_individuals, max_layers, dm, max_parameters=100000):
        self.dm = dm # Data module for model creation
        
        self.n_individuals = n_individuals
        self.max_layers = max_layers
        self.generation = 0
        self.max_parameters = max_parameters
        self.save_directory = "./models_traced"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = self.setup_logger()
        
    
    @staticmethod
    def setup_logger(log_file='./logs/population.log', log_level=logging.DEBUG):
        """
        Set up a logger for the population module.

        If the log file already exists, create a new one by appending a timestamp to the filename.

        Parameters:
            log_file (str): Path to the log file.
            log_level (int): Logging level (e.g., logging.DEBUG, logging.INFO).

        Returns:
            logging.Logger: Configured logger instance.
        """
        if os.path.exists(log_file):
            base, ext = os.path.splitext(log_file)
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{base}_{timestamp}{ext}"

        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)
        # Create file handler and set level to debug
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # Add formatter to file handler
        file_handler.setFormatter(formatter)
        # Add file handler to logger
        logger.addHandler(file_handler)
        return logger
    
        
    def initial_poll(self):
        """
        Generate the initial population of individuals.    
        """
        
        self.population = self.create_population()
        self._checkpoint()


    def create_random_individual(self):
        """
        Create a random individual with a random number of layers.
        """
        return Individual(max_layers=self.max_layers)
    

    def sort_population(self):
        """
        Sort the population by fitness.
        """
        self.population = sorted(self.population, key=lambda individual: individual.fitness, reverse=True)
        self.checkpoint()
        

    def _checkpoint(self):
        """
        Save the current population.
        """
        os.makedirs(self.save_directory, exist_ok=True)
        self._update_df()
        self.save_population()
        self.save_dataframe()
    
    
    def check_individual(self, individual):
        try:
            model_representation, is_valid = self.build_model(individual.parsed_layers)
            if is_valid:
                modelSize = self.evaluate_parameters(model_representation)
                individual.model_size = modelSize
                
                assert modelSize > 0, f"Model size must be greater then zero: {modelSize} Parameters"
                assert modelSize < self.max_parameters, f"Model size is too big: {modelSize} Parameters"
                assert modelSize is not None, f"Model size is None..."
                return True # Individual is valid
        except Exception as e:
                self.logger.error(f"Error encountered when checking individual: {e}")
                return False # Individual is invalid


    def create_population(self):
        """
        Create a population of unique, valid individuals.

        This function generates random individuals one by one and checks if they are valid using check_individual.
        After each candidate is generated, duplicates are removed using remove_duplicates until the population
        size reaches n_individuals.

        Returns:
            list: A list of unique, valid individuals.
        """
        population = []
        with tqdm.tqdm(total=self.n_individuals, desc="Generating Population") as pbar:
            # Generate individuals until the population reaches n_individuals, removing duplicates along the way
            while len(population) < self.n_individuals:
                candidate = self.create_random_individual()  # Create a random individual
                if self.check_individual(candidate):
                    population.append(candidate)
                    pbar.update(1)  # Update the progress bar for each valid individual
            
            population = self.remove_duplicates(population)  # Remove duplicates
        return population


    def elite_models(self, k_best=1):
        """
        Retrieve the top k_best elite models from the current population based on fitness.

        The population is sorted in descending order based on the fitness attribute of each individual.
        This function then returns deep copies of the top k_best individuals to ensure that the
        original models remain immutable during further operations.

        Parameters:
            k_best (int): The number of top-performing individuals to retrieve. Defaults to 1.

        Returns:
            list: A list containing deep copies of the elite individuals.
        """
        sorted_pop = sorted(self, key=lambda individual: individual.fitness, reverse=True)
        topModels = [deepcopy(sorted_pop[i]) for i in range(k_best)]
        return topModels


    def evolve(self, mating_pool_cutoff=0.5, mutation_probability=0.85, k_best=1, n_random=3):
        """
        Generates a new population ensuring that the total number of individuals equals pop.n_individuals.
        
        Parameters:
            pop                  : List or collection of individuals. Assumed to have attributes: 
                                .n_individuals and .generation.
            mating_pool_cutoff   : Fraction determining the size of the mating pool (top percent of individuals).
            mutation_probability : The probability to use during mutation.
            k_best               : The number of best individuals from the current population to retain.
        
        Returns:
            new_population: A list representing the new generation of individuals.
            
        Note:
            Assumes that helper functions single_point_crossover(), mutation(), and create_random_individual() exist.
        """
        new_population = []
        self.generation += 1
        self.topModels = self.elite_models(k_best=k_best)


        # 2. Create the mating pool based on the cutoff from the sorted population
        sorted_pop = sorted(self, key=lambda individual: individual.fitness, reverse=True)
        mating_pool = sorted_pop[:int(np.floor(mating_pool_cutoff * self.n_individuals))].copy()
        assert len(mating_pool) > 0, "Mating pool is empty."
        
        # Generate offspring until reaching the desired population size
        while len(new_population) < self.n_individuals - n_random - k_best:
            try:
                parent1 = np.random.choice(mating_pool)
                parent2 = np.random.choice(mating_pool)
                assert parent1.parsed_layers != parent2.parsed_layers, "Parents are the same individual."
            except Exception as e:
                self.logger.error(f"Error selecting parents: {e}")
                continue
            
            # a) Crossover:
            children = single_point_crossover([parent1, parent2])
            # b) Mutation:
            mutated_children = gene_mutation(children, mutation_probability)
            # c) Random choice of one of the mutated children
            for kid in mutated_children:
                kid.reset()
                if self.check_individual(kid):
                    new_population.append(kid)
                else:
                    pass


        # 3. Add random individuals to the new population
        while len(new_population) < self.n_individuals - k_best:
            try:
                individual = self.create_random_individual()
                model_representation, is_valid = self.build_model(individual.parsed_layers)
                if is_valid:
                    individual.model_size = int(self.evaluate_parameters(model_representation))
                    assert individual.model_size > 0, f"Model size is {individual.model_size}"
                    assert individual.model_size < self.max_parameters, f"Model size is {individual.model_size}"
                    assert individual.model_size is not None, f"Model size is None"
                    new_population.append(individual)
            except Exception as e:
                self.logger.error(f"Error encountered when evolving population: {e}")
                continue
        
        
        # 4. Add the best individuals from the previous generation
        new_population.extend(self.topModels)
       

        assert len(new_population) == self.n_individuals, f"Population size is {len(new_population)}, expected {self.n_individuals}"
        self.population = new_population
        self._checkpoint()


    def remove_duplicates(self, population):
        """
        Remove duplicates from the given population by replacing duplicates with newly generated unique individuals.

        Parameters:
            population (list): A list of individuals in the population.

        Returns:
            list: The updated population with duplicates removed.
        """
        unique_architectures = set()
        updated_population = []

        for individual in population:
            # Use the 'architecture' attribute if available, otherwise fallback to a default representation.
            arch = getattr(individual, 'architecture', None)
            if arch is None:
                # If no architecture attribute, use parsed_layers as unique identifier.
                arch = str(individual.parsed_layers)

            if arch not in unique_architectures:
                unique_architectures.add(arch)
                updated_population.append(individual)
            else:
                # Try to generate a unique individual up to 50 times
                for _ in range(50):
                    new_individual = Individual(max_layers=self.max_layers)
                    new_arch = getattr(new_individual, 'architecture', None)
                    if new_arch is None:
                        new_arch = str(new_individual.parsed_layers)

                    if new_arch not in unique_architectures:
                        unique_architectures.add(new_arch)
                        updated_population.append(new_individual)
                        break
                else:
                    # After 50 attempts, keep the original duplicate as a fallback.
                    updated_population.append(individual)
        return updated_population
        
    
    def build_model(self, parsed_layers, task="segmentation"):
        """
        Build a model based on the provided parsed layers.

        This function creates an encoder using the parsed layers and constructs a model by combining
        the encoder with a head layer via the ModelConstructor. The constructed model is built to
        process inputs defined by the data module (dm).

        Parameters:
            parsed_layers: The parsed architecture configuration used by the encoder to build the network.

        Returns:
            A PyTorch model constructed with the encoder and head layer.
        """
        
        def shape_tracer(self, encoder):
            """
            Traces the output shapes of a given encoder model when provided with a dummy input.
            Args:
                encoder (torch.nn.Module): The encoder model whose output shapes are to be traced.
            Returns:
                list[tuple]: A list of tuples representing the shapes of the encoder's outputs 
                     (excluding the batch dimension). If the encoder outputs a single tensor, 
                     the list will contain one tuple. If the encoder outputs multiple tensors 
                     (e.g., a list or tuple of tensors), the list will contain a tuple for each output.
            """
            
            dummy_input = torch.randn(1, *self.dm.input_shape).to(self.device)
            with torch.no_grad():
                output = encoder(dummy_input)
            shapes = []
            if isinstance(output, (list, tuple)):
                for o in output:
                    shape_without_batch = tuple(o.shape[1:])
                    shapes.append(shape_without_batch)
            else:
                shape_without_batch = tuple(output.shape[1:])
                shapes.append(shape_without_batch)
            return shapes
        
        self.task = task
        
        if task == "segmentation":
            model = GenericUNetNetwork(parsed_layers,
                    input_channels=self.dm.input_shape[0], 
                    input_height=self.dm.input_shape[1], 
                    input_width=self.dm.input_shape[2], 
                    num_classes=self.dm.num_classes,
                    encoder_only=False,
            )
            valid = True
        elif task == "classification": 
            encoder = GenericUNetNetwork(parsed_layers,
                    input_channels=self.dm.input_shape[0], 
                    input_height=self.dm.input_shape[1], 
                    input_width=self.dm.input_shape[2], 
                    num_classes=self.dm.num_classes,
                    encoder_only=True,
            )
            valid = True
                
            head = MultiInputClassifier(shape_tracer(self, encoder.to(self.device)), num_classes=self.dm.num_classes)
            head = head.to(self.device)
            model = nn.Sequential(encoder, head)
            
        else:
            raise ValueError(f"Task {task} not supported.")
        
        return model, valid
    
    
    def evaluate_parameters(self, model):
        """
        Calculate the total number of parameters of the given model.

        Parameters:
            model (torch.nn.Module): The PyTorch model.

        Returns:
            int: The total number of parameters.
        """
        num_params = sum(p.numel() for p in model.parameters())
        return num_params
    
    
    def _update_df(self):
        """
        Create a DataFrame from the population.

        Returns:
            pd.DataFrame: A DataFrame containing the population.
        """
        columns = ["Generation", "Layers", "Fitness", "Metric", "FPS", "Params"]
        data = []
        for individual in self.population:
            generation = self.generation
            parsed_layers = individual.parsed_layers
            metric = individual.metric
            fps = individual.fps
            fitness = individual.fitness
            model_size = individual.model_size
            data.append([generation, parsed_layers, fitness, metric, fps, model_size])
        
        df = pd.DataFrame(data, columns=columns).sort_values(by="Fitness", ascending=False)
        df.reset_index(drop=True, inplace=True)
        
        self.df = df
    
    
    def save_dataframe(self):
        """
        Save the DataFrame containing the population statistics to a pickle file.

        The DataFrame is saved at a path that includes the current generation number.
        In case of an error during saving, the exception details are printed.

        Returns:
            None
        """
        path = f'{self.save_directory}/src/df_population_{self.generation}.pkl'
        try:
            self.df.to_pickle(path)
            self.logger.info(f"DataFrame saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving DataFrame to {path}: {e}")
    
    
    def load_dataframe(self, generation):
        path = f'./models_traced/src/df_population_{generation}.pkl'
        try:
            df = pd.read_pickle(path)
            return df
        except Exception as e:
            self.logger.error(f"Error loading DataFrame from {path}: {e}")
            return None
    
    
    def save_population(self):
        path = f'./models_traced/src/population_{self.generation}.pkl'
        try:
            with open(path, 'wb') as f:
                pickle.dump(self.population, f)
            self.logger.info(f"Population saved to {path}")
        except Exception as e:
            self.logger.error(f"Error saving population to {path}: {e}")
    
    
    def load_population(self, generation):
        path = f'./models_traced/src/population_{generation}.pkl'
        try:
            with open(path, 'rb') as f:
                population = pickle.load(f)
            return population
        except Exception as e:
            self.logger.error(f"Error loading population from {path}: {e}")
            return None
    

    def train_individual(self, idx, task, epochs=20, lr=1e-3, batch_size=None):
        """
        Train the individual using the data module and the specified number of epochs and learning rate.

        Parameters:
            individual (Individual): The individual to train.
            epochs (int): The number of epochs to train the individual. Defaults to 20.
            lr (float): The learning rate to use during training. Defaults to 1e-3.

        Returns:
            None
        """
        individual = self.population[idx]
        
        
        model, _ = self.build_model(individual.parsed_layers, task=task)
        if task == "segmentation":
            LM = GenericLightningSegmentationNetwork(
                model=model,
                learning_rate=lr,
            )
        
        elif task == "classification":
            LM = GenericLightningNetwork(
                model=model,
                learning_rate=lr,
                num_classes=self.dm.num_classes,
            )
        else:
            raise ValueError(f"Task {task} not supported.")
        
        # Create a PyTorch Lightning trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu"
        )
        # Set the batch size if specified
        if batch_size is not None:
            self.dm.batch_size = batch_size
        # Train the lightning model
        trainer.fit(LM, self.dm)
        results = trainer.test(LM, self.dm)
        individual._prompt_fitness(results[0])
        self._checkpoint()        


    def train_generation(self, task='classification', lr=0.001, epochs=4, batch_size=32):
        """
        Train all individuals in the current generation that have not been trained yet.

        Parameters:
            task (str): The task type ('classification' or 'segmentation').
            lr (float): Learning rate for training.
            epochs (int): Number of epochs for training.
            batch_size (int): Batch size for training.

        Returns:
            None
        """
        for idx in range(len(self)):
            if 'Fitness' in self.df.columns and not pd.isna(self.df.loc[idx, 'Fitness']) and self.df.loc[idx, 'Fitness'] != 0:
                print(f"Skipping individual {idx}/{len(self)} as it has already been trained")
                continue

            print(f"Training individual {idx}/{len(self)}")
            self.train_individual(idx=idx, task=task, lr=lr, epochs=epochs, batch_size=batch_size)
            clear_output(wait=True)

        
    def save_model(self, LM,
                   save_torchscript=True, 
                   ts_save_path=None,
                   save_standard=True, 
                   std_save_path=None):
        # Use generation attribute from the Population object.
        gen = self.population.generation
        
        if ts_save_path is None:
            ts_save_path = f"models_traced/generation_{gen}/model_and_architecture_{self.idx}.pt"
        if std_save_path is None:
            std_save_path = f"models_traced/generation_{gen}/model_{self.idx}.pth"
        
        # Save the results to a text file.
        with open(f"models_traced/generation_{gen}/results_model_{self.idx}.txt", "w") as f:
            f.write("Test Results:\n")
            for key, value in self.results[0].items():
                f.write(f"{key}: {value}\n")
        
        # Prepare dummy input from dm.input_shape
        input_shape = self.dm.input_shape
        if len(input_shape) == 3:
            input_shape = (1,) + input_shape
        device = next(LM.parameters()).device
        example_input = torch.randn(*input_shape).to(device)
        
        LM = LM.eval()  # set the model to evaluation mode
        
        if save_torchscript:
            traced_model = torch.jit.trace(LM.model, example_input)
            traced_model.save(ts_save_path)
            print(f"Scripted (TorchScript) model saved at {ts_save_path}")
        
        if save_standard:
            # Retrieve architecture code from the individual.
            arch_code = self.population[self.idx].architecture
            save_dict = {"state_dict": LM.model.state_dict()}
            if arch_code is not None:
                save_dict["architecture_code"] = arch_code
            torch.save(save_dict, std_save_path)
            print(f"Standard model saved at {std_save_path}")


    def __getitem__(self, index):
        """
        Retrieve an individual from the population at the specified index.

        Args:
            index (int): The index of the individual to retrieve.

        Returns:
            object: The individual at the specified index in the population.
        """
        return self.population[index]


    def __len__(self):
        """
        Returns the number of individuals in the population.

        This method allows the use of the `len()` function to retrieve
        the size of the population.

        Returns:
            int: The number of individuals in the population.
        """
        return len(self.population)  
