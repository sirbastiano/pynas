import configparser
import pandas as pd

import torch
import pytorch_lightning as pl

from pynas.core.population import Population
from datasets.Phisat2SimulatedData.segmentation_dataset import SegmentationDataModule

import argparse

# Define dataset module
root_dir = '/Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/data/Phisat2Simulation'
dm = SegmentationDataModule(root_dir, batch_size=8, num_workers=1, transform=None, val_split=0.3)

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
    epochs = int(config['GA']['epochs'])
    batch_size = int(config['GA']['batch_size'])
    n_random = int(config['GA']['n_random'])
    k_best = int(config['GA']['k_best'])
    task = str(config['GA']['task'])
    
    
    # Define population
    pop = Population(n_individuals=n_individuals, max_layers=max_layers, dm=dm, max_parameters=400_000)
    # if you want to use group norm in the decoder, set the following to True
    pop._use_group_norm = False
    
    if args.gen is not None:
        pop.load_generation(args.gen)
    else:
        pop.initial_poll()

    for _ in range(max_gen):
        pop.train_generation(task=task, lr=0.001, epochs=epochs, batch_size=batch_size)
        pop.evolve(mating_pool_cutoff=mating_pool_cutoff, mutation_probability=mutation_probability, k_best=k_best, n_random=n_random)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args=args)