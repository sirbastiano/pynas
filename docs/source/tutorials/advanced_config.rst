Advanced Configuration
=====================

This tutorial covers advanced configuration options for PyNAS, including parameter tuning, custom search spaces, and optimization strategies.

Configuration Files
-------------------

PyNAS supports comprehensive configuration through INI files. Here's how to create and use advanced configurations:

Creating a Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a ``config.ini`` file with the following structure:

.. code-block:: ini

    [Task]
    ; Task type: segmentation or classification
    type_head=segmentation
    
    [NAS]
    ; Architecture search parameters
    max_layers=7
    
    [GA]
    ; Genetic Algorithm parameters
    population_size=50
    epochs=15
    batch_size=16
    max_iterations=25
    mating_pool_cutoff=0.6
    mutation_probability=0.25
    n_random=8
    k_best=3
    task=segmentation
    
    [Computation]
    ; Hardware and computation settings
    seed=42
    num_workers=8
    accelerator=gpu
    
    [Optimizer]
    ; Optimizer selection (1=GreyWolf, 2=ParticleSwarm)
    optimizer_selection=1

Loading and Using Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    batch_size = config.getint('GA', 'batch_size')
    mutation_prob = config.getfloat('GA', 'mutation_probability')
    
    # Use in population creation
    pop = Population(
        n_individuals=population_size,
        max_layers=max_layers,
        dm=dm,
        max_parameters=500_000
    )

Custom Search Spaces
--------------------

PyNAS allows you to define custom search spaces for different layer types.

Configuring Layer Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify the configuration to customize layer parameter ranges:

.. code-block:: ini

    [ConvAct]
    ; Convolutional layer parameters
    min_kernel_size = 3
    max_kernel_size = 7
    min_stride = 1
    max_stride = 2
    min_padding = 1
    max_padding = 3
    min_out_channels_coefficient = 4
    max_out_channels_coefficient = 16
    
    [MBConv]
    ; MobileNet Inverted Bottleneck parameters
    min_expansion_factor = 2
    max_expansion_factor = 8
    min_dw_kernel_size = 3
    max_dw_kernel_size = 5
    
    [ResNetBlock]
    ; ResNet block parameters
    min_reduction_factor = 2
    max_reduction_factor = 6

Custom Fitness Functions
------------------------

Create custom fitness evaluation strategies:

.. code-block:: python

    from pynas.train.myFit import FitnessEvaluator
    
    class CustomFitnessEvaluator(FitnessEvaluator):
        """Custom fitness evaluator with domain-specific metrics."""
        
        def weighted_sum_exponential(self, fps: float, iou: float) -> float:
            """
            Custom fitness function emphasizing accuracy over speed.
            
            Args:
                fps: Frames per second (inference speed)
                iou: Intersection over Union (accuracy metric)
                
            Returns:
                Fitness score combining speed and accuracy
            """
            if fps is None or iou is None:
                return 0.0
                
            # Custom weights favoring accuracy
            accuracy_weight = 0.8
            speed_weight = 0.2
            
            # Normalize metrics
            normalized_iou = min(iou, 1.0)
            normalized_fps = min(fps / 100.0, 1.0)  # Assume max 100 FPS
            
            # Exponential weighting for accuracy
            fitness = (accuracy_weight * (normalized_iou ** 2) + 
                      speed_weight * normalized_fps)
            
            return fitness
    
    # Use custom evaluator
    evaluator = CustomFitnessEvaluator()

Multi-Objective Optimization
----------------------------

Balance multiple objectives like accuracy, speed, and model size:

.. code-block:: python

    def pareto_fitness(individual):
        """
        Multi-objective fitness using Pareto ranking.
        
        Args:
            individual: Individual to evaluate
            
        Returns:
            Pareto rank and crowding distance
        """
        objectives = [
            individual.iou,           # Maximize accuracy
            individual.fps,           # Maximize speed
            -individual.model_size    # Minimize model size
        ]
        
        # Implement Pareto ranking logic
        return calculate_pareto_rank(objectives)
    
    # Apply during evolution
    def custom_evolution_step(pop):
        """Custom evolution with Pareto selection."""
        # Calculate Pareto ranks for all individuals
        for individual in pop.population:
            individual.pareto_rank = pareto_fitness(individual)
        
        # Select based on Pareto dominance
        selected = pareto_selection(pop.population, pop.n_individuals // 2)
        return selected

Advanced Evolution Strategies
-----------------------------

Implement sophisticated evolution techniques:

Adaptive Mutation Rates
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class AdaptiveEvolution:
        """Evolution with adaptive parameters."""
        
        def __init__(self, initial_mutation_rate=0.2):
            self.mutation_rate = initial_mutation_rate
            self.generation = 0
            self.best_fitness_history = []
        
        def adapt_mutation_rate(self, current_best_fitness):
            """Adapt mutation rate based on progress."""
            if len(self.best_fitness_history) > 5:
                recent_improvement = (current_best_fitness - 
                                    self.best_fitness_history[-5])
                
                if recent_improvement < 0.01:  # Stagnation
                    self.mutation_rate = min(0.5, self.mutation_rate * 1.2)
                else:  # Good progress
                    self.mutation_rate = max(0.1, self.mutation_rate * 0.9)
            
            self.best_fitness_history.append(current_best_fitness)
        
        def evolve_generation(self, pop):
            """Evolve with adaptive parameters."""
            # Get current best fitness
            pop._sort_population()
            best_fitness = pop.population[0].fitness
            
            # Adapt mutation rate
            self.adapt_mutation_rate(best_fitness)
            
            # Evolve with current mutation rate
            pop.evolve(
                mutation_probability=self.mutation_rate,
                mating_pool_cutoff=0.5,
                k_best=3,
                n_random=5
            )
            
            print(f"Generation {self.generation}: "
                  f"Best fitness = {best_fitness:.4f}, "
                  f"Mutation rate = {self.mutation_rate:.3f}")
            
            self.generation += 1

Island Evolution
~~~~~~~~~~~~~~~

Run parallel evolution with migration:

.. code-block:: python

    def island_evolution(dm, num_islands=4, migration_interval=5):
        """
        Run evolution on multiple islands with periodic migration.
        
        Args:
            dm: Data module
            num_islands: Number of parallel populations
            migration_interval: Generations between migrations
        """
        # Create multiple populations (islands)
        islands = []
        for i in range(num_islands):
            pop = Population(
                n_individuals=20,
                max_layers=5,
                dm=dm,
                max_parameters=300_000
            )
            pop.initial_poll()
            islands.append(pop)
        
        # Evolution with migration
        for generation in range(30):
            # Evolve each island
            for i, island in enumerate(islands):
                print(f"Evolving island {i+1}")
                island.train_generation(task='classification', epochs=8)
                island._sort_population()
                
                if generation < 29:  # Don't evolve on last generation
                    island.evolve()
            
            # Migration every few generations
            if generation % migration_interval == 0 and generation > 0:
                migrate_individuals(islands, num_migrants=2)
        
        # Combine results from all islands
        all_individuals = []
        for island in islands:
            all_individuals.extend(island.population)
        
        # Sort and return best
        all_individuals.sort(key=lambda x: x.fitness, reverse=True)
        return all_individuals[:10]

Performance Monitoring
---------------------

Advanced monitoring and analysis tools:

.. code-block:: python

    import matplotlib.pyplot as plt
    import pandas as pd
    
    class EvolutionAnalyzer:
        """Analyze and visualize evolution progress."""
        
        def __init__(self):
            self.generation_stats = []
        
        def record_generation(self, pop, generation):
            """Record statistics for current generation."""
            if not pop.population:
                return
                
            fitnesses = [ind.fitness for ind in pop.population if ind.fitness]
            
            stats = {
                'generation': generation,
                'best_fitness': max(fitnesses) if fitnesses else 0,
                'mean_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else 0,
                'std_fitness': pd.Series(fitnesses).std() if fitnesses else 0,
                'population_size': len(pop.population)
            }
            
            self.generation_stats.append(stats)
        
        def plot_evolution(self):
            """Plot evolution progress."""
            df = pd.DataFrame(self.generation_stats)
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(df['generation'], df['best_fitness'], 'b-', linewidth=2)
            plt.title('Best Fitness Over Generations')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            
            plt.subplot(2, 2, 2)
            plt.plot(df['generation'], df['mean_fitness'], 'g-', linewidth=2)
            plt.title('Mean Fitness Over Generations')
            plt.xlabel('Generation')
            plt.ylabel('Mean Fitness')
            
            plt.subplot(2, 2, 3)
            plt.plot(df['generation'], df['std_fitness'], 'r-', linewidth=2)
            plt.title('Fitness Standard Deviation')
            plt.xlabel('Generation')
            plt.ylabel('Std Deviation')
            
            plt.tight_layout()
            plt.show()
    
    # Usage
    analyzer = EvolutionAnalyzer()
    
    for generation in range(20):
        # ... evolution code ...
        analyzer.record_generation(pop, generation)
    
    analyzer.plot_evolution()

Hyperparameter Optimization
---------------------------

Optimize evolution hyperparameters:

.. code-block:: python

    from sklearn.model_selection import ParameterGrid
    
    def hyperparameter_search(dm):
        """Search for optimal evolution hyperparameters."""
        
        param_grid = {
            'population_size': [20, 30, 50],
            'mutation_probability': [0.1, 0.2, 0.3],
            'mating_pool_cutoff': [0.4, 0.5, 0.6],
            'k_best': [2, 3, 5]
        }
        
        best_params = None
        best_result = 0
        
        for params in ParameterGrid(param_grid):
            print(f"Testing parameters: {params}")
            
            # Create population with current parameters
            pop = Population(
                n_individuals=params['population_size'],
                max_layers=5,
                dm=dm,
                max_parameters=200_000
            )
            
            # Run short evolution
            pop.initial_poll()
            for gen in range(5):  # Short test
                pop.train_generation(task='classification', epochs=5)
                pop._sort_population()
                if gen < 4:
                    pop.evolve(
                        mutation_probability=params['mutation_probability'],
                        mating_pool_cutoff=params['mating_pool_cutoff'],
                        k_best=params['k_best']
                    )
            
            # Evaluate result
            final_fitness = pop.population[0].fitness if pop.population else 0
            
            if final_fitness > best_result:
                best_result = final_fitness
                best_params = params
        
        print(f"Best parameters: {best_params}")
        print(f"Best result: {best_result}")
        return best_params

Next Steps
----------

- Experiment with different configuration combinations
- Implement custom architectural components
- Try multi-objective optimization for your specific use case
- Explore distributed evolution for larger search spaces
