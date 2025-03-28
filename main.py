import configparser
import pandas as pd

import torch
import pytorch_lightning as pl

from pynas.core.population import Population
from datasets.RawClassifier.loader import RawClassifierDataModule
import argparse

# Define dataset module
root_dir = '/Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/data/RawClassifier'
dm = RawClassifierDataModule(root_dir, batch_size=4, num_workers=2, transform=None)

pd.set_option('display.max_colwidth', None)

# Argument parser
parser = argparse.ArgumentParser(description="Run the PyNAS genetic algorithm for neural architecture search.")
parser.add_argument('--gen', type=int, default=None, help='Path to the configuration file.')
parser.add_argument('--config', type=str, default='config.ini', help='Path to the configuration file.')



def main(args):
    config = configparser.ConfigParser()
    config.read(args.config)
    seed = config.getint(section='Computation', option='seed')
    pl.seed_everything(seed=seed, workers=True)  # For reproducibility
    torch.set_float32_matmul_precision("medium")  # to make lightning happy
    
    # Model parameters
    max_layers = int(config['NAS']['max_layers'])
    max_gen = int(config['GA']['max_iterations'])
    n_individuals = int(config['GA']['population_size'])
    mating_pool_cutoff = float(config['GA']['mating_pool_cutoff'])
    mutation_probability = float(config['GA']['mutation_probability'])
    
    # Define population
    pop = Population(n_individuals=n_individuals, max_layers=max_layers, dm=dm, max_parameters=400_000)
    if args.gen is not None:
        pop.load_generation(args.gen)
    else:
        pop.initial_poll()

    for _ in range(max_gen):
        pop.train_generation(task='classification', lr=0.001, epochs=15, batch_size=16)
        pop.evolve(mating_pool_cutoff=mating_pool_cutoff, mutation_probability=mutation_probability, k_best=1, n_random=3)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args=args)