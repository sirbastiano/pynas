Vessel Detection Example
=======================

This example demonstrates how to use PyNAS for automated vessel detection in Synthetic Aperture Radar (SAR) imagery, a common application in maritime surveillance and remote sensing.

Overview
--------

Vessel detection in SAR images presents unique challenges:

- **Speckle noise** inherent in SAR imagery
- **Variable vessel sizes** from small boats to large ships
- **Complex backgrounds** including sea clutter and weather effects
- **Real-time requirements** for operational systems

PyNAS can automatically discover architectures optimized for these specific challenges while maintaining efficiency for edge deployment.

Dataset Setup
-------------

First, set up the vessel detection dataset:

.. code-block:: python

    import os
    import torch
    import pytorch_lightning as pl
    from datasets.RawVessels.loader import RawVesselsDataModule
    
    # Set up reproducibility
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")
    
    # Configure dataset paths
    root_dir = 'data/TASI/DataSAR_real_refined'
    
    # Verify dataset exists
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Dataset not found at {root_dir}")
    
    # Initialize data module for vessel detection
    dm = RawVesselsDataModule(
        root_dir=root_dir,
        batch_size=8,           # Adjust based on GPU memory
        num_workers=4,          # Parallel data loading
        test_size=0.15,         # 15% for testing
        val_size=0.15,          # 15% for validation
        seed=42,                # Reproducibility
        transform=None          # Add augmentations if needed
    )
    
    # Setup the data module
    dm.setup()
    
    # Inspect dataset properties
    print(f"Dataset information:")
    print(f"  Input shape: {dm.input_shape}")
    print(f"  Number of classes: {dm.num_classes}")
    print(f"  Training samples: {len(dm.train_dataset)}")
    print(f"  Validation samples: {len(dm.val_dataset)}")
    print(f"  Test samples: {len(dm.test_dataset)}")

Population Configuration
------------------------

Configure the genetic algorithm for vessel detection:

.. code-block:: python

    from pynas.core.population import Population
    
    # Create population optimized for vessel detection
    pop = Population(
        n_individuals=30,        # Moderate population size
        max_layers=6,           # Allow deeper networks for complex features
        dm=dm,                  # Vessel detection data module
        max_parameters=300_000  # Balance accuracy and efficiency
    )
    
    print(f"Initialized population for vessel detection:")
    print(f"  Population size: {pop.n_individuals}")
    print(f"  Max layers: {pop.max_layers}")
    print(f"  Parameter budget: {pop.max_parameters:,}")

Architecture Search Process
---------------------------

Run the neural architecture search optimized for vessel detection:

.. code-block:: python

    import configparser
    
    # Load vessel-specific configuration
    config = configparser.ConfigParser()
    config.read('pynas/core/config.ini')
    
    # Extract vessel detection parameters
    epochs = 12              # More epochs for complex SAR data
    batch_size = 8           # Balance memory and gradient quality
    max_generations = 15     # Sufficient exploration
    
    # Initialize population
    print("Generating initial population for vessel detection...")
    pop.initial_poll()
    
    # Evolution loop
    best_fitness_history = []
    
    for generation in range(max_generations):
        print(f"\\n{'='*50}")
        print(f"Generation {generation + 1}/{max_generations}")
        print(f"{'='*50}")
        
        # Train current generation
        print("Training architectures on vessel detection task...")
        pop.train_generation(
            task='classification',  # Binary: vessel/no-vessel
            epochs=epochs,
            lr=0.001,              # Conservative learning rate
            batch_size=batch_size
        )
        
        # Sort population by fitness
        pop._sort_population()
        
        # Record best fitness
        if pop.population:
            best_fitness = pop.population[0].fitness
            best_fitness_history.append(best_fitness)
            
            print(f"Best fitness: {best_fitness:.4f}")
            print(f"Best IoU: {pop.population[0].iou:.4f}")
            print(f"Best FPS: {pop.population[0].fps:.2f}")
            print(f"Best model size: {pop.population[0].model_size:,} parameters")
        
        # Evolve to next generation (except last)
        if generation < max_generations - 1:
            print("Evolving to next generation...")
            pop.evolve(
                mating_pool_cutoff=0.6,    # Top 60% for mating
                mutation_probability=0.25,  # Higher mutation for exploration
                k_best=3,                   # Keep top 3 performers
                n_random=5                  # Add diversity
            )
        
        # Checkpoint progress
        pop.save_population()
        print(f"Generation {generation + 1} completed and saved")

Results Analysis
----------------

Analyze the discovered architectures:

.. code-block:: python

    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Get top performing models
    top_models = pop.elite_models(k_best=5)
    
    print("\\n" + "="*60)
    print("VESSEL DETECTION RESULTS")
    print("="*60)
    
    for i, model in enumerate(top_models):
        print(f"\\nRank {i+1}:")
        print(f"  Architecture: {model.architecture}")
        print(f"  Fitness: {model.fitness:.4f}")
        print(f"  Detection Accuracy (IoU): {model.iou:.4f}")
        print(f"  Inference Speed: {model.fps:.2f} FPS")
        print(f"  Model Complexity: {model.model_size:,} parameters")
        
        # Calculate efficiency ratio
        efficiency = model.iou / (model.model_size / 1000)  # Accuracy per 1K params
        print(f"  Efficiency Ratio: {efficiency:.6f} IoU/1K params")
    
    # Plot evolution progress
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(best_fitness_history) + 1), best_fitness_history, 'b-o')
    plt.title('Fitness Evolution - Vessel Detection')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    
    # Fitness vs Model Size scatter plot
    plt.subplot(2, 2, 2)
    fitnesses = [m.fitness for m in top_models]
    sizes = [m.model_size for m in top_models]
    plt.scatter(sizes, fitnesses, c='red', alpha=0.7, s=100)
    plt.title('Fitness vs Model Size')
    plt.xlabel('Model Size (parameters)')
    plt.ylabel('Fitness')
    plt.grid(True)
    
    # IoU vs FPS trade-off
    plt.subplot(2, 2, 3)
    ious = [m.iou for m in top_models]
    fps = [m.fps for m in top_models]
    plt.scatter(fps, ious, c='green', alpha=0.7, s=100)
    plt.title('Accuracy vs Speed Trade-off')
    plt.xlabel('Inference Speed (FPS)')
    plt.ylabel('Detection Accuracy (IoU)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

Model Validation
----------------

Validate the best model on test data:

.. code-block:: python

    # Get the best performing model
    best_individual = pop.population[0]
    
    # Build the final model
    best_model, is_valid = pop.build_model(
        best_individual.parsed_layers,
        task='classification'
    )
    
    if is_valid:
        print(f"\\nBest model validation:")
        print(f"  Architecture: {best_individual.architecture}")
        print(f"  Model built successfully: {is_valid}")
        print(f"  Total parameters: {pop.evaluate_parameters(best_model):,}")
        
        # Additional validation metrics
        print(f"\\nPerformance metrics:")
        print(f"  Final test accuracy: {best_individual.iou:.4f}")
        print(f"  Inference speed: {best_individual.fps:.2f} FPS")
        print(f"  Memory footprint: {best_individual.model_size:,} parameters")
        
        # Calculate deployment metrics
        model_size_mb = best_individual.model_size * 4 / (1024**2)  # Assuming float32
        print(f"  Estimated model size: {model_size_mb:.2f} MB")
        
        # Real-time capability assessment
        if best_individual.fps >= 10:
            print("  ✅ Suitable for real-time vessel detection")
        else:
            print("  ⚠️  May not meet real-time requirements")

Deployment Preparation
---------------------

Prepare the model for edge deployment:

.. code-block:: python

    import torch.jit
    
    # Export best model for deployment
    best_model.eval()
    
    # Create example input for tracing
    example_input = torch.randn(1, *dm.input_shape)
    
    # Trace the model for deployment
    traced_model = torch.jit.trace(best_model, example_input)
    
    # Save the traced model
    model_path = f"vessel_detection_best_model.pt"
    traced_model.save(model_path)
    print(f"Model saved for deployment: {model_path}")
    
    # Save architecture information
    architecture_info = {
        'architecture_code': best_individual.architecture,
        'parsed_layers': best_individual.parsed_layers,
        'fitness': best_individual.fitness,
        'iou': best_individual.iou,
        'fps': best_individual.fps,
        'model_size': best_individual.model_size,
        'input_shape': dm.input_shape,
        'num_classes': dm.num_classes
    }
    
    import json
    with open('vessel_detection_architecture.json', 'w') as f:
        json.dump(architecture_info, f, indent=2)
    
    print("Architecture information saved: vessel_detection_architecture.json")

Performance Comparison
---------------------

Compare with baseline architectures:

.. code-block:: python

    # Define baseline architectures for comparison
    baselines = {
        'Simple CNN': 'c3g c3g c3g a C',
        'ResNet-like': 'c3g R3g R3g a C', 
        'MobileNet-like': 'm4g m4g m4g a C'
    }
    
    print("\\nComparing with baseline architectures:")
    print("-" * 60)
    
    for name, arch_code in baselines.items():
        # Create individual with baseline architecture
        from pynas.core.individual import Individual
        baseline_individual = Individual(max_layers=5)
        baseline_individual.architecture = arch_code
        
        # Parse and validate
        from pynas.core import architecture_builder
        baseline_individual.parsed_layers = architecture_builder.parse_architecture_code(arch_code)
        
        if pop.check_individual(baseline_individual):
            print(f"{name}:")
            print(f"  Architecture: {arch_code}")
            print(f"  Model size: {baseline_individual.model_size:,} parameters")
            # Note: Would need to train for fair comparison
        else:
            print(f"{name}: Invalid architecture")

Key Findings
------------

From this vessel detection example, typical findings include:

1. **Architecture Patterns**: Successful architectures often combine:
   - Multiple convolutional layers for feature extraction
   - Efficient blocks (MBConv) for parameter efficiency  
   - Appropriate pooling for spatial reduction

2. **Performance Trade-offs**:
   - Higher accuracy often requires more parameters
   - Real-time performance favors efficient architectures
   - SAR-specific features benefit from deeper networks

3. **Optimization Insights**:
   - Genetic algorithms effectively explore architecture space
   - Population diversity is crucial for finding optimal solutions
   - Evolution converges to practical architectures

Next Steps
----------

- Experiment with data augmentation for improved robustness
- Try multi-scale architectures for detecting vessels of different sizes
- Implement ensemble methods combining top architectures
- Explore transfer learning from pre-trained models
