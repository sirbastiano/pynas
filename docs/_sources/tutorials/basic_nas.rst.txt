Basic Neural Architecture Search
================================

This tutorial covers the fundamental concepts and workflow of neural architecture search using PyNAS.

Overview
--------

Neural Architecture Search (NAS) automates the design of neural network architectures. PyNAS uses genetic algorithms to evolve populations of neural networks, selecting the best performing ones based on defined fitness criteria.

Key Concepts
------------

**Individual**
    A single neural network architecture represented by an encoded string

**Population** 
    A collection of individuals that evolve over generations

**Fitness**
    A metric combining performance (IoU/accuracy) and efficiency (FPS/latency)

**Generation**
    One complete cycle of evaluation, selection, and evolution

Step 1: Setting Up Data
-----------------------

First, prepare your dataset using PyNAS data modules:

.. code-block:: python

    from datasets.RawVessels.loader import RawVesselsDataModule
    
    # Configure your dataset
    root_dir = 'data/TASI/DataSAR_real_refined'
    dm = RawVesselsDataModule(
        root_dir=root_dir,
        batch_size=8,
        num_workers=4,
        test_size=0.15,
        val_size=0.15,
        seed=42
    )
    
    # Setup the data module
    dm.setup()
    
    # Inspect dataset properties
    print(f"Input shape: {dm.input_shape}")
    print(f"Number of classes: {dm.num_classes}")

Step 2: Creating a Population
-----------------------------

Initialize a population with your desired parameters:

.. code-block:: python

    from pynas.core.population import Population
    import pytorch_lightning as pl
    
    # Set random seed for reproducibility
    pl.seed_everything(42, workers=True)
    
    # Create population
    pop = Population(
        n_individuals=20,        # Population size
        max_layers=5,           # Maximum layers per architecture
        dm=dm,                  # Data module
        max_parameters=200_000  # Parameter budget constraint
    )
    
    print(f"Created population with {pop.n_individuals} individuals")

Step 3: Initial Population Generation
-------------------------------------

Generate the initial random population:

.. code-block:: python

    # Generate initial population
    print("Generating initial population...")
    pop.initial_poll()
    
    # Inspect some individuals
    for i, individual in enumerate(pop.population[:3]):
        print(f"Individual {i}:")
        print(f"  Architecture: {individual.architecture}")
        print(f"  Model size: {individual.model_size} parameters")

Step 4: Training and Evaluation
-------------------------------

Train the current generation:

.. code-block:: python

    # Train the generation
    print("Training generation...")
    pop.train_generation(
        task='classification',  # or 'segmentation'
        epochs=10,             # Training epochs per individual
        lr=0.001,              # Learning rate
        batch_size=8           # Batch size
    )
    
    # Sort population by fitness
    pop._sort_population()
    
    # Display results
    print("Top 3 individuals after training:")
    for i in range(min(3, len(pop.population))):
        individual = pop.population[i]
        print(f"Rank {i+1}:")
        print(f"  Fitness: {individual.fitness:.4f}")
        print(f"  IoU: {individual.iou:.4f}")
        print(f"  FPS: {individual.fps:.2f}")

Step 5: Evolution Process
-------------------------

Evolve the population over multiple generations:

.. code-block:: python

    max_generations = 10
    
    for generation in range(max_generations):
        print(f"\\n=== Generation {generation + 1} ===")
        
        # Train current generation
        pop.train_generation(
            task='classification',
            epochs=8,
            lr=0.001,
            batch_size=8
        )
        
        # Sort and get best fitness
        pop._sort_population()
        best_fitness = pop.population[0].fitness if pop.population else 0
        print(f"Best fitness: {best_fitness:.4f}")
        
        # Evolve to next generation (except last iteration)
        if generation < max_generations - 1:
            pop.evolve(
                mating_pool_cutoff=0.5,    # Top 50% for mating
                mutation_probability=0.2,   # 20% mutation rate
                k_best=2,                   # Keep 2 best individuals
                n_random=3                  # Add 3 random individuals
            )
            
        # Save progress
        pop.save_population()

Step 6: Analyzing Results
-------------------------

Extract and analyze the best architectures:

.. code-block:: python

    # Get elite models
    top_models = pop.elite_models(k_best=5)
    
    print("\\n=== Final Results ===")
    for i, model in enumerate(top_models):
        print(f"\\nModel {i+1}:")
        print(f"  Architecture: {model.architecture}")
        print(f"  Fitness: {model.fitness:.4f}")
        print(f"  IoU: {model.iou:.4f}")
        print(f"  FPS: {model.fps:.2f}")
        print(f"  Parameters: {model.model_size:,}")
        
    # Save results to DataFrame
    pop.save_dataframe()
    print("\\nResults saved to dataframe")

Step 7: Model Export
--------------------

Export the best model for deployment:

.. code-block:: python

    # Get the best individual
    best_individual = pop.population[0]
    
    # Build the final model
    model, is_valid = pop.build_model(
        best_individual.parsed_layers, 
        task='classification'
    )
    
    if is_valid:
        print("Best model built successfully!")
        print(f"Model parameters: {pop.evaluate_parameters(model):,}")
        
        # The model can now be saved or deployed
        # torch.save(model.state_dict(), 'best_model.pth')

Understanding the Output
------------------------

**Fitness Score**
    Combines accuracy/IoU and inference speed using a weighted function

**IoU (Intersection over Union)**
    Segmentation metric or classification accuracy depending on task

**FPS (Frames Per Second)**
    Inference speed measurement for deployment considerations

**Model Size**
    Total number of trainable parameters

Next Steps
----------

- Try different population sizes and evolution parameters
- Experiment with custom fitness functions
- Explore different architectural building blocks
- Learn about :doc:`advanced_config` for fine-tuning
