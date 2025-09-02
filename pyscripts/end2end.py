# The code is designed to be run from the command line with var# Define dataset module
import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # To ignore any user warnings from libraries

import argparse, os, sys
# Get the parent directory of the script (project root)
if '__file__' in globals():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)
else:
    # Fallback for interactive environments
    sys.path.insert(0, '..')

import configparser
import pandas as pd
import torch
import pytorch_lightning as pl

# --------- Custom imports ---------
from pynas.core.population import Population
from datasets.RawClassifier.loader import ClassifierDataModule
# --------- Custom imports ---------
# Set pandas options for better display
pd.set_option('display.max_colwidth', None)
cwd = os.getcwd()




# =========== Data Checks ===========
# Check if the dataset directory exists
root_dir = os.path.join(project_root, 'data', 'end2end')
if not os.path.exists(root_dir):
    print(f"Dataset path {root_dir} does not exist. Please check the path or download the dataset.")
    sys.exit(1)



# =========================================
# Argument parser
parser = argparse.ArgumentParser(description="Run the PyNAS genetic algorithm for neural architecture search.")
parser.add_argument('--gen', type=int, default=None, help='Generation to load and start NAS.')
parser.add_argument('--config', type=str, default='config.ini', help='Path to the configuration file.')
# Computation parameters
parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
# NAS parameters
parser.add_argument('--max_layers', type=int, default=None, help='Maximum number of layers in the architecture.')
parser.add_argument('--max_parameters', type=int, default=None, help='Maximum number of parameters allowed in models.')
# GA parameters
parser.add_argument('--max_iterations', type=int, default=None, help='Maximum number of generations.')
parser.add_argument('--population_size', type=int, default=None, help='Size of the population.')
parser.add_argument('--mating_pool_cutoff', type=float, default=None, help='Fraction of population to use for mating.')
parser.add_argument('--mutation_probability', type=float, default=None, help='Probability of mutation.')
parser.add_argument('--epochs', type=int, default=None, help='Number of epochs for training each model.')
parser.add_argument('--batch_size', type=int, default=None, help='Batch size for training.')
parser.add_argument('--n_random', type=int, default=None, help='Number of random individuals per generation.')
parser.add_argument('--k_best', type=int, default=None, help='Number of best individuals to keep.')
parser.add_argument('--task', type=str, default=None, help='Task type.')
# Dataset parameters
parser.add_argument('--dataset', type=str, default='end2end', help='Dataset to use for training. Default is TASI.')
# =========================================



# Define dataset module
dm = ClassifierDataModule(root_dir, batch_size=8, num_workers=0, transform=None, 
                    mode='semisplit' )

# =========== Data Checks ===========
# Add this debugging code before creating the population
print("=== Data Module Debug Info ===")
print(f"Input shape: {dm.input_shape}")
print(f"Number of classes: {dm.num_classes}")
print(f"Data module type: {type(dm)}")





def main(args):
    try:
        config = configparser.ConfigParser()
        config.read(args.config)
        
        # Override config with command line arguments if provided
        if args.seed is not None:
            config.set('Computation', 'seed', str(args.seed))
        if args.max_layers is not None:
            config.set('NAS', 'max_layers', str(args.max_layers))
        if args.max_parameters is not None:
            config.set('GA', 'max_parameters', str(args.max_parameters))
        if args.max_iterations is not None:
            config.set('GA', 'max_iterations', str(args.max_iterations))
        if args.population_size is not None:
            config.set('GA', 'population_size', str(args.population_size))
        if args.mating_pool_cutoff is not None:
            config.set('GA', 'mating_pool_cutoff', str(args.mating_pool_cutoff))
        if args.mutation_probability is not None:
            config.set('GA', 'mutation_probability', str(args.mutation_probability))
        if args.epochs is not None:
            config.set('GA', 'epochs', str(args.epochs))
        if args.batch_size is not None:
            config.set('GA', 'batch_size', str(args.batch_size))
        if args.n_random is not None:
            config.set('GA', 'n_random', str(args.n_random))
        if args.k_best is not None:
            config.set('GA', 'k_best', str(args.k_best))
        if args.task is not None:
            config.set('GA', 'task', args.task)
        
        # Read updated configuration
        seed = config.getint(section='Computation', option='seed')
        pl.seed_everything(seed=seed, workers=True)  # For reproducibility
        torch.set_float32_matmul_precision("medium")  # to make lightning happy
        save_dir = os.path.join(cwd, 'Results')
        os.makedirs(save_dir, exist_ok=True)
        
        # Model parameters
        max_layers = int(config['NAS']['max_layers'])
        max_gen = int(config['GA']['max_iterations'])
        n_individuals = int(config['GA']['population_size'])
        mating_pool_cutoff = float(config['GA']['mating_pool_cutoff'])
        mutation_probability = float(config['GA']['mutation_probability'])
        epochs = int(config['GA']['epochs'])
        batch_size = int(config['GA']['batch_size'])
        n_random = int(config['GA']['n_random'])
        k_best = int(config['GA']['k_best'])
        task = str(config['GA']['task'])
        max_params = int(config['GA']['max_parameters'])
        

        
        # 0. Define population
        pop = Population(n_individuals=n_individuals, 
                        max_layers=max_layers, 
                        dm=dm,
                        save_directory=save_dir,
                        max_parameters=max_params)
        # TODO: if you want to use group norm in the decoder, set the following to True
        pop._use_group_norm = False
        
        # 1. Copy config.ini to the results directory
        config_path = os.path.join(pop.save_directory, 'config.ini')
        if not os.path.exists(config_path):
            with open(config_path, 'w') as configfile:
                config.write(configfile)
        

        
        if args.gen is not None:
            pop.load_population(args.gen) # load a generation from the saved models
            print(f"Generation {args.gen} loaded.")
        else:
            pop.initial_poll() # create a new generation
            
        # 2. Train and evolve the population
        start_gen = args.gen if args.gen is not None else 0
        for gen in range(start_gen, max_gen):
            print(f"=== Processing Generation {gen} ===")
            
            # Check if generation needs training
            needs_training = pop.check_generation_needs_training()
            if needs_training:
                print(f"Training generation {gen}...")
                pop.train_generation(task=task, lr=0.001, epochs=epochs, batch_size=batch_size)
            else:
                print(f"Generation {gen} already trained, skipping training.")
            
            # Only evolve if not the last generation
            if gen < max_gen - 1:
                print(f"Evolving generation {gen} to {gen + 1}...")
                pop.evolve(mating_pool_cutoff=mating_pool_cutoff, mutation_probability=mutation_probability, k_best=k_best, n_random=n_random)

        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    
if __name__ == '__main__':
    args = parser.parse_args()
    r = main(args=args)
    if r == 0:
        print("Execution completed successfully.")
    else:
        print("Execution failed.")
    sys.exit(r)
