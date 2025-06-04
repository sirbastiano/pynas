Quick Start Guide
=================

This guide will help you get started with PyNAS for neural architecture search using genetic algorithms.

Basic Concepts
--------------

PyNAS implements neural architecture search through genetic algorithms with the following key components:

- **Individual**: Represents a single neural network architecture
- **Population**: A collection of individuals (architectures) 
- **Evolution**: The process of improving architectures through genetic operations
- **Fitness**: A measure of how well an architecture performs on a given task

First Example: Basic Usage
---------------------------

Here's a minimal example to get you started:

.. code-block:: python

   import configparser
   import torch
   import pytorch_lightning as pl
   
   from pynas.core.population import Population
   from datasets.RawVessels.loader import RawVesselsDataModule
   
   # Set up reproducibility
   pl.seed_everything(42, workers=True)
   torch.set_float32_matmul_precision("medium")
   
   # Configure data module
   root_dir = 'data/TASI/DataSAR_real_refined'
   dm = RawVesselsDataModule(
       root_dir=root_dir,
       batch_size=4,
       num_workers=2,
       transform=None
   )
   
   # Create a population for genetic algorithm
   pop = Population(
       n_individuals=10,      # Small population for testing
       max_layers=5,          # Maximum layers per architecture
       dm=dm,                 # Data module
       max_parameters=100_000 # Parameter budget constraint
   )
   
   # Initialize the population
   print("Creating initial population...")
   pop.initial_poll()
   
   # Train one generation
   print("Training generation...")
   pop.train_generation(
       task='classification',
       epochs=5,              # Few epochs for quick testing
       lr=0.001,
       batch_size=4
   )

Configuration-Based Approach
-----------------------------

PyNAS supports configuration files for more complex setups:

1. Create a configuration file (``config.ini``):

.. code-block:: ini

   [GA]
   population_size=50
   epochs=10
   batch_size=8
   max_iterations=20
   mating_pool_cutoff=0.5
   mutation_probability=0.2
   n_random=5
   k_best=2
   task=segmentation
   
   [NAS]
   max_layers=5
   
   [Computation]
   seed=42
   num_workers=8

2. Use the configuration in your script:

.. code-block:: python

   import configparser
   from pynas.core.population import Population
   
   # Load configuration
   config = configparser.ConfigParser()
   config.read('config.ini')
   
   # Extract parameters
   population_size = config.getint('GA', 'population_size')
   max_layers = config.getint('NAS', 'max_layers')
   epochs = config.getint('GA', 'epochs')
   
   # Create population with config parameters
   pop = Population(
       n_individuals=population_size,
       max_layers=max_layers,
       dm=dm,
       max_parameters=400_000
   )

Creating and Evaluating Individuals
------------------------------------

You can also work with individual architectures directly:

.. code-block:: python

   from pynas.core.individual import Individual
   
   # Create a random individual
   individual = Individual(max_layers=5, min_layers=3)
   
   # View the architecture
   print(f"Architecture: {individual.architecture}")
   print(f"Parsed layers: {individual.parsed_layers}")
   
   # Check if individual is valid
   is_valid = pop.check_individual(individual)
   print(f"Valid architecture: {is_valid}")

Evolution Process
-----------------

Run a complete evolutionary process:

.. code-block:: python

   # Configuration for evolution
   max_generations = 10
   
   for generation in range(max_generations):
       print(f"\\n=== Generation {generation + 1} ===")
       
       # Train current generation
       pop.train_generation(
           task='classification',
           epochs=5,
           lr=0.001,
           batch_size=4
       )
       
       # Sort population by fitness
       pop._sort_population()
       
       # Print best fitness
       if pop.population:
           best_fitness = pop.population[0].fitness
           print(f"Best fitness: {best_fitness}")
       
       # Evolve to next generation (except for last iteration)
       if generation < max_generations - 1:
           pop.evolve(
               mating_pool_cutoff=0.5,
               mutation_probability=0.2,
               k_best=2,
               n_random=3
           )

Monitoring Progress
-------------------

PyNAS provides built-in logging and checkpointing:

.. code-block:: python

   # Population automatically logs to './logs/population.log'
   # Check logs for detailed information about:
   # - Population creation progress
   # - Training metrics
   # - Evolution statistics
   # - Error handling
   
   # Save population state
   pop.save_population()
   
   # Load previous generation
   loaded_pop = pop.load_population(generation=5)

Working with Results
--------------------

Access and analyze the results:

.. code-block:: python

   # Get the best performing architectures
   top_models = pop.elite_models(k_best=3)
   
   for i, model in enumerate(top_models):
       print(f"Model {i+1}:")
       print(f"  Architecture: {model.architecture}")
       print(f"  Fitness: {model.fitness}")
       print(f"  IoU: {model.iou}")
       print(f"  FPS: {model.fps}")
       print(f"  Model size: {model.model_size} parameters")

Next Steps
----------

- Explore the :doc:`tutorials/index` for more detailed examples
- Check the :doc:`api/core` for complete API documentation  
- See :doc:`examples/index` for real-world use cases
- Learn about custom :doc:`api/blocks` for extending architectures
