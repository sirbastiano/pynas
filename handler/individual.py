import architecture_builder as builder
import torch ##

class Individual:
    def __init__(self, max_layers, min_layers=3):
        self.architecture = builder.generate_random_architecture_code(max_layers=max_layers, min_layers=min_layers)
        self.chromosome = self.architecture2chromosome(input_architecture=self.architecture)
        self.parsed_layers = builder.parse_architecture_code(self.architecture)
        self.reset()

    def __str__(self):
        return f'Individual: {self.architecture}'
    
    
    def _reparse_layers(self):
        self.parsed_layers = builder.parse_architecture_code(self.chromosome2architecture(self.chromosome))


    def reset(self):
        """
        Resets the individual's fitness, IOU, FPS, and model size.
        """
        self.fitness = 0.0
        self.iou = None
        self.fps = None
        self.model_size = None
        self.model = None


    def architecture2chromosome(self, input_architecture):
        """
        Converts an architecture code into a chromosome list by splitting
        the architecture code using 'E'. This method also handles the case where
        the architecture ends with 'EE', avoiding an empty string at the end of the list.
        """
        # Split the architecture code on 'E'
        chromosome = input_architecture.split('E')
        # Remove the last two empty elements if the architecture ends with 'EE'
        if len(chromosome) >= 2 and chromosome[-1] == '' and chromosome[-2] == '':
            chromosome = chromosome[:-2]
        elif len(chromosome) >= 1 and chromosome[-1] == '':
            # If it only ends with a single 'E', just remove the last empty element
            chromosome = chromosome[:-1]
        return chromosome

    def chromosome2architecture(self, input_chromosome):
        """
        Converts the chromosome list back into an architecture code by joining
        the list items with 'E' and ensuring the architecture ends with 'EE'.
        """
        architecture_code = 'E'.join(input_chromosome) + 'EE'
        return architecture_code

    def copy(self):
        """
        Creates a deep copy of the current individual, including architecture,
        chromosome, and fitness.
        """
        new_individual = Individual(max_layers=len(self.chromosome))
        new_individual.architecture = self.architecture
        new_individual.chromosome = self.chromosome.copy()
        new_individual.fitness = self.fitness
        new_individual.iou = self.iou
        new_individual.fps = self.fps
        new_individual.model_size = self.model_size
        
        if self.model is not None:
            new_individual.model = self.model  # Copy the entire model

        return new_individual    
    
    def set_trained_model(self, model):
        """
        Set the trained model.
        """
        self.model = model

