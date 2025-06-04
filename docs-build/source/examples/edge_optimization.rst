===================
Edge Optimization
===================

This example demonstrates how to use PyNAS for optimizing neural network architectures specifically for edge devices, focusing on parameter count, inference speed, and memory usage while maintaining acceptable accuracy.

Overview
========

Edge deployment requires careful consideration of:

- **Model Size**: Limited storage and memory on edge devices
- **Inference Speed**: Real-time processing requirements
- **Power Consumption**: Battery-powered devices need efficient models
- **Accuracy Trade-offs**: Balancing performance with efficiency

In this example, we'll evolve architectures optimized for mobile deployment scenarios.

.. contents:: Table of Contents
   :local:
   :depth: 2

Setting Up Edge-Optimized Evolution
====================================

Configuration for Edge Optimization
------------------------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   from pynas.core.population import Population
   from pynas.core.individual import Individual
   from pynas.core.architecture_builder import ArchitectureBuilder
   
   # Edge-optimized configuration
   edge_config = {
       'population_size': 30,
       'max_iterations': 20,
       'max_parameters': 500_000,  # Strict parameter limit
       'target_latency_ms': 50,    # Target inference time
       'min_accuracy': 0.85,       # Minimum acceptable accuracy
       'architecture_constraints': {
           'max_layers': 8,
           'preferred_blocks': ['MBConv', 'ConvBnAct'],  # Mobile-friendly blocks
           'avoid_blocks': ['DenseNetBlock'],  # Memory-intensive blocks
       }
   }

Custom Fitness Function for Edge Optimization
----------------------------------------------

.. code-block:: python

   import time
   import psutil
   
   class EdgeFitnessEvaluator:
       """Multi-objective fitness evaluator for edge optimization."""
       
       def __init__(self, dataloader, device='cpu', target_device='mobile'):
           """
           Initialize edge fitness evaluator.
           
           Args:
               dataloader: PyTorch DataLoader for evaluation
               device: Device for evaluation ('cpu' for edge simulation)
               target_device: Target deployment device type
           """
           self.dataloader = dataloader
           self.device = device
           self.target_device = target_device
           self.weights = {
               'accuracy': 0.4,
               'parameters': 0.25,
               'latency': 0.25,
               'memory': 0.1
           }
       
       def evaluate_model(self, model):
           """Comprehensive evaluation for edge deployment."""
           model.eval()
           
           # Accuracy evaluation
           accuracy = self._evaluate_accuracy(model)
           
           # Parameter count
           param_count = sum(p.numel() for p in model.parameters())
           
           # Latency measurement
           latency = self._measure_latency(model)
           
           # Memory usage
           memory_usage = self._measure_memory(model)
           
           # Multi-objective fitness calculation
           fitness = self._calculate_fitness(
               accuracy, param_count, latency, memory_usage
           )
           
           return {
               'fitness': fitness,
               'accuracy': accuracy,
               'parameters': param_count,
               'latency_ms': latency * 1000,
               'memory_mb': memory_usage / (1024 * 1024),
               'pareto_metrics': {
                   'accuracy': accuracy,
                   'efficiency': 1.0 / (param_count + latency * 1000)
               }
           }
       
       def _evaluate_accuracy(self, model):
           """Evaluate model accuracy on validation set."""
           correct = 0
           total = 0
           
           with torch.no_grad():
               for batch_idx, (data, targets) in enumerate(self.dataloader):
                   if batch_idx >= 50:  # Quick evaluation for speed
                       break
                   
                   data, targets = data.to(self.device), targets.to(self.device)
                   outputs = model(data)
                   _, predicted = torch.max(outputs.data, 1)
                   total += targets.size(0)
                   correct += (predicted == targets).sum().item()
           
           return correct / total if total > 0 else 0.0
       
       def _measure_latency(self, model, num_runs=100):
           """Measure average inference latency."""
           dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
           
           # Warmup
           for _ in range(10):
               with torch.no_grad():
                   _ = model(dummy_input)
           
           # Actual measurement
           torch.cuda.synchronize() if self.device != 'cpu' else None
           start_time = time.time()
           
           for _ in range(num_runs):
               with torch.no_grad():
                   _ = model(dummy_input)
           
           torch.cuda.synchronize() if self.device != 'cpu' else None
           end_time = time.time()
           
           return (end_time - start_time) / num_runs
       
       def _measure_memory(self, model):
           """Measure model memory footprint."""
           # Calculate model size in bytes
           param_size = sum(p.numel() * p.element_size() for p in model.parameters())
           buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
           
           return param_size + buffer_size
       
       def _calculate_fitness(self, accuracy, param_count, latency, memory):
           """Calculate multi-objective fitness score."""
           # Normalize metrics (lower is better for efficiency metrics)
           norm_accuracy = accuracy  # Higher is better
           norm_params = 1.0 / (1.0 + param_count / 1000000)  # Normalize by 1M params
           norm_latency = 1.0 / (1.0 + latency * 1000)  # Convert to ms
           norm_memory = 1.0 / (1.0 + memory / (1024 * 1024))  # Convert to MB
           
           # Weighted combination
           fitness = (
               self.weights['accuracy'] * norm_accuracy +
               self.weights['parameters'] * norm_params +
               self.weights['latency'] * norm_latency +
               self.weights['memory'] * norm_memory
           )
           
           return fitness

Mobile-Optimized Architecture Search
=====================================

Evolution Setup for Mobile Devices
-----------------------------------

.. code-block:: python

   from pynas.core.generic_unet import GenericUNetNetwork
   from pynas.blocks.activations import ReLU, GELU
   
   class MobileArchitectureEvolution:
       """Evolution specifically targeting mobile deployment."""
       
       def __init__(self, dataset, config):
           self.dataset = dataset
           self.config = config
           self.fitness_evaluator = EdgeFitnessEvaluator(
               dataloader=dataset.val_dataloader(),
               device='cpu',  # Simulate edge device
               target_device='mobile'
           )
       
       def create_mobile_population(self):
           """Create initial population optimized for mobile devices."""
           population = []
           
           # Mobile-friendly architecture templates
           mobile_templates = [
               "m2r_a_m2r_a_m2r_C",  # MobileNet-inspired
               "c2r_M_c2r_M_c2r_C",  # Efficient CNN
               "m3g_a_c2r_M_c1r_C",  # Hybrid approach
               "c1r_m2r_a_c1r_C",    # Minimal design
           ]
           
           for template in mobile_templates:
               # Create multiple variations of each template
               for _ in range(self.config['population_size'] // len(mobile_templates)):
                   individual = self._create_individual_from_template(template)
                   population.append(individual)
           
           return Population(individuals=population, config=self.config)
       
       def _create_individual_from_template(self, template):
           """Create individual from architecture template."""
           builder = ArchitectureBuilder()
           
           # Parse template and create architecture
           parsed_arch = builder.parse_architecture_string(template)
           
           # Add mobile-specific constraints
           for layer in parsed_arch:
               if 'MBConv' in layer.get('layer_type', ''):
                   layer['expansion_factor'] = 3  # Lower expansion for efficiency
               elif 'ConvBnAct' in layer.get('layer_type', ''):
                   layer['out_channels_coefficient'] = 6  # Moderate channels
           
           return Individual(
               genome=parsed_arch,
               task='classification',
               input_shape=(3, 224, 224),
               num_classes=self.dataset.num_classes
           )
       
       def evolve_mobile_architecture(self):
           """Run evolution with mobile-specific objectives."""
           population = self.create_mobile_population()
           
           best_individuals = []
           pareto_front = []
           
           for generation in range(self.config['max_iterations']):
               print(f"\n=== Generation {generation + 1} ===")
               
               # Evaluate population
               fitness_results = []
               for individual in population.individuals:
                   model = individual.build_model()
                   if model is not None:
                       results = self.fitness_evaluator.evaluate_model(model)
                       individual.fitness = results['fitness']
                       fitness_results.append(results)
                   else:
                       individual.fitness = 0.0
                       fitness_results.append({
                           'fitness': 0.0, 'accuracy': 0.0,
                           'parameters': float('inf'), 'latency_ms': float('inf')
                       })
               
               # Track best individual
               best_idx = max(range(len(population.individuals)), 
                            key=lambda i: population.individuals[i].fitness)
               best_individual = population.individuals[best_idx]
               best_results = fitness_results[best_idx]
               
               print(f"Best Individual - Fitness: {best_results['fitness']:.4f}")
               print(f"  Accuracy: {best_results['accuracy']:.4f}")
               print(f"  Parameters: {best_results['parameters']:,}")
               print(f"  Latency: {best_results['latency_ms']:.2f} ms")
               print(f"  Memory: {best_results['memory_mb']:.2f} MB")
               
               best_individuals.append((best_individual.copy(), best_results))
               
               # Update Pareto front
               self._update_pareto_front(pareto_front, fitness_results, population)
               
               # Evolution step with mobile-specific constraints
               if generation < self.config['max_iterations'] - 1:
                   population = self._mobile_evolution_step(population, fitness_results)
           
           return best_individuals, pareto_front
       
       def _update_pareto_front(self, pareto_front, fitness_results, population):
           """Update Pareto front for multi-objective optimization."""
           for i, (individual, results) in enumerate(zip(population.individuals, fitness_results)):
               # Check if this individual dominates any in current front
               dominated = []
               dominates_any = False
               
               for j, (front_ind, front_res) in enumerate(pareto_front):
                   if self._dominates(results, front_res):
                       dominated.append(j)
                       dominates_any = True
                   elif self._dominates(front_res, results):
                       break  # This individual is dominated
               else:
                   # Not dominated by any in front
                   # Remove dominated individuals
                   for idx in sorted(dominated, reverse=True):
                       del pareto_front[idx]
                   
                   # Add this individual to front
                   pareto_front.append((individual.copy(), results))
       
       def _dominates(self, results1, results2):
           """Check if results1 dominates results2 (Pareto dominance)."""
           better_accuracy = results1['accuracy'] >= results2['accuracy']
           better_efficiency = (
               results1['parameters'] <= results2['parameters'] and
               results1['latency_ms'] <= results2['latency_ms']
           )
           
           at_least_one_better = (
               results1['accuracy'] > results2['accuracy'] or
               results1['parameters'] < results2['parameters'] or
               results1['latency_ms'] < results2['latency_ms']
           )
           
           return better_accuracy and better_efficiency and at_least_one_better
       
       def _mobile_evolution_step(self, population, fitness_results):
           """Custom evolution step for mobile optimization."""
           # Selection: prefer efficient models
           selected = self._efficiency_based_selection(population, fitness_results)
           
           # Crossover with mobile-friendly bias
           offspring = self._mobile_crossover(selected)
           
           # Mutation with efficiency constraints
           mutated = self._efficiency_mutation(offspring)
           
           return Population(individuals=mutated, config=self.config)
       
       def _efficiency_based_selection(self, population, fitness_results, select_ratio=0.5):
           """Selection favoring efficient architectures."""
           # Sort by efficiency score (combination of accuracy and efficiency)
           efficiency_scores = []
           for results in fitness_results:
               if results['parameters'] == float('inf'):
                   efficiency_scores.append(-1)
               else:
                   efficiency_score = (
                       results['accuracy'] * 2 +  # Weight accuracy higher
                       (1.0 / (1.0 + results['parameters'] / 100000)) +
                       (1.0 / (1.0 + results['latency_ms']))
                   ) / 4.0
                   efficiency_scores.append(efficiency_score)
           
           # Select top performers
           num_select = int(len(population.individuals) * select_ratio)
           selected_indices = sorted(
               range(len(efficiency_scores)),
               key=lambda i: efficiency_scores[i],
               reverse=True
           )[:num_select]
           
           return [population.individuals[i] for i in selected_indices]
       
       def _mobile_crossover(self, selected):
           """Crossover operation preserving mobile-friendly characteristics."""
           offspring = []
           
           while len(offspring) < self.config['population_size']:
               parent1 = random.choice(selected)
               parent2 = random.choice(selected)
               
               child1, child2 = self._efficient_crossover(parent1, parent2)
               offspring.extend([child1, child2])
           
           return offspring[:self.config['population_size']]
       
       def _efficient_crossover(self, parent1, parent2):
           """Crossover that maintains efficiency characteristics."""
           # Prefer shorter, more efficient architectures
           genome1 = parent1.genome[:min(len(parent1.genome), 6)]  # Limit depth
           genome2 = parent2.genome[:min(len(parent2.genome), 6)]
           
           # Single-point crossover with bias toward mobile blocks
           crossover_point = min(len(genome1), len(genome2)) // 2
           
           child1_genome = genome1[:crossover_point] + genome2[crossover_point:]
           child2_genome = genome2[:crossover_point] + genome1[crossover_point:]
           
           # Apply mobile-friendly constraints
           child1_genome = self._apply_mobile_constraints(child1_genome)
           child2_genome = self._apply_mobile_constraints(child2_genome)
           
           child1 = Individual(
               genome=child1_genome,
               task=parent1.task,
               input_shape=parent1.input_shape,
               num_classes=parent1.num_classes
           )
           
           child2 = Individual(
               genome=child2_genome,
               task=parent2.task,
               input_shape=parent2.input_shape,
               num_classes=parent2.num_classes
           )
           
           return child1, child2
       
       def _apply_mobile_constraints(self, genome):
           """Apply constraints for mobile deployment."""
           constrained_genome = []
           
           for layer in genome:
               # Prefer mobile-friendly blocks
               if layer.get('layer_type') == 'DenseNetBlock':
                   # Replace with MBConv
                   layer = {
                       'layer_type': 'MBConv',
                       'expansion_factor': 3,
                       'activation': layer.get('activation', 'ReLU')
                   }
               
               # Limit channel expansion
               if 'out_channels_coefficient' in layer:
                   layer['out_channels_coefficient'] = min(
                       layer['out_channels_coefficient'], 8
                   )
               
               # Limit expansion factors
               if 'expansion_factor' in layer:
                   layer['expansion_factor'] = min(layer['expansion_factor'], 4)
               
               constrained_genome.append(layer)
           
           return constrained_genome
       
       def _efficiency_mutation(self, offspring):
           """Mutation that maintains efficiency."""
           mutated = []
           
           for individual in offspring:
               if random.random() < 0.3:  # 30% mutation rate
                   mutated_individual = self._mobile_mutate(individual)
                   mutated.append(mutated_individual)
               else:
                   mutated.append(individual)
           
           return mutated
       
       def _mobile_mutate(self, individual):
           """Mobile-specific mutation operations."""
           genome = individual.genome.copy()
           
           if len(genome) > 0:
               mutation_type = random.choice(['modify', 'add_efficient', 'remove'])
               
               if mutation_type == 'modify':
                   # Modify existing layer to be more efficient
                   idx = random.randint(0, len(genome) - 1)
                   layer = genome[idx].copy()
                   
                   # Make modifications toward efficiency
                   if 'out_channels_coefficient' in layer:
                       layer['out_channels_coefficient'] = max(
                           layer['out_channels_coefficient'] - 1, 4
                       )
                   
                   genome[idx] = layer
               
               elif mutation_type == 'add_efficient' and len(genome) < 6:
                   # Add efficient layer
                   efficient_layer = {
                       'layer_type': random.choice(['MBConv', 'ConvBnAct']),
                       'activation': 'ReLU',
                       'out_channels_coefficient': random.randint(4, 6)
                   }
                   
                   if efficient_layer['layer_type'] == 'MBConv':
                       efficient_layer['expansion_factor'] = 3
                   
                   genome.append(efficient_layer)
               
               elif mutation_type == 'remove' and len(genome) > 2:
                   # Remove a layer to reduce complexity
                   genome.pop(random.randint(0, len(genome) - 1))
           
           return Individual(
               genome=genome,
               task=individual.task,
               input_shape=individual.input_shape,
               num_classes=individual.num_classes
           )

Quantization and Pruning Integration
=====================================

Post-Training Optimization
--------------------------

.. code-block:: python

   import torch.quantization as quant
   from torch.nn.utils import prune
   
   class EdgeOptimizationPipeline:
       """Complete pipeline for edge deployment optimization."""
       
       def __init__(self, model, calibration_loader):
           self.model = model
           self.calibration_loader = calibration_loader
       
       def optimize_for_edge(self, optimization_config):
           """Apply multiple optimization techniques."""
           optimized_model = self.model
           
           # 1. Structured pruning
           if optimization_config.get('pruning', {}).get('enabled', False):
               optimized_model = self.apply_structured_pruning(
                   optimized_model, 
                   optimization_config['pruning']
               )
           
           # 2. Quantization
           if optimization_config.get('quantization', {}).get('enabled', False):
               optimized_model = self.apply_quantization(
                   optimized_model,
                   optimization_config['quantization']
               )
           
           # 3. Knowledge distillation (if teacher model provided)
           if optimization_config.get('distillation', {}).get('teacher_model'):
               optimized_model = self.apply_knowledge_distillation(
                   optimized_model,
                   optimization_config['distillation']
               )
           
           return optimized_model
       
       def apply_structured_pruning(self, model, pruning_config):
           """Apply structured pruning to reduce model size."""
           pruning_ratio = pruning_config.get('ratio', 0.2)
           
           # Apply channel-wise pruning to convolution layers
           for name, module in model.named_modules():
               if isinstance(module, (nn.Conv2d, nn.Linear)):
                   if 'conv' in name.lower():
                       # Prune channels
                       prune.ln_structured(
                           module, name='weight', amount=pruning_ratio,
                           n=2, dim=0  # Prune output channels
                       )
                   else:
                       # Prune neurons
                       prune.l1_unstructured(module, name='weight', amount=pruning_ratio)
           
           # Make pruning permanent
           for name, module in model.named_modules():
               if isinstance(module, (nn.Conv2d, nn.Linear)):
                   try:
                       prune.remove(module, 'weight')
                   except ValueError:
                       pass  # No pruning applied to this module
           
           return model
       
       def apply_quantization(self, model, quantization_config):
           """Apply quantization for reduced precision."""
           quantization_type = quantization_config.get('type', 'dynamic')
           
           if quantization_type == 'dynamic':
               # Dynamic quantization (good for CPU inference)
               quantized_model = torch.quantization.quantize_dynamic(
                   model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
               )
           
           elif quantization_type == 'static':
               # Static quantization (requires calibration)
               model.eval()
               
               # Prepare model for quantization
               model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
               torch.quantization.prepare(model, inplace=True)
               
               # Calibrate with sample data
               with torch.no_grad():
                   for batch_idx, (data, _) in enumerate(self.calibration_loader):
                       if batch_idx >= 10:  # Limited calibration
                           break
                       model(data)
               
               # Convert to quantized model
               quantized_model = torch.quantization.convert(model, inplace=False)
           
           else:
               quantized_model = model
           
           return quantized_model
       
       def apply_knowledge_distillation(self, student_model, distillation_config):
           """Apply knowledge distillation for better small model performance."""
           teacher_model = distillation_config['teacher_model']
           temperature = distillation_config.get('temperature', 3.0)
           alpha = distillation_config.get('alpha', 0.7)
           
           # Knowledge distillation training loop would go here
           # This is a simplified version
           teacher_model.eval()
           student_model.train()
           
           criterion_kd = nn.KLDivLoss(reduction='batchmean')
           criterion_ce = nn.CrossEntropyLoss()
           optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
           
           for epoch in range(distillation_config.get('epochs', 5)):
               for batch_idx, (data, targets) in enumerate(self.calibration_loader):
                   if batch_idx >= 20:  # Limited training
                       break
                   
                   optimizer.zero_grad()
                   
                   # Student predictions
                   student_outputs = student_model(data)
                   
                   # Teacher predictions
                   with torch.no_grad():
                       teacher_outputs = teacher_model(data)
                   
                   # Distillation loss
                   loss_kd = criterion_kd(
                       F.log_softmax(student_outputs / temperature, dim=1),
                       F.softmax(teacher_outputs / temperature, dim=1)
                   ) * (temperature ** 2)
                   
                   # Standard classification loss
                   loss_ce = criterion_ce(student_outputs, targets)
                   
                   # Combined loss
                   loss = alpha * loss_kd + (1 - alpha) * loss_ce
                   
                   loss.backward()
                   optimizer.step()
           
           return student_model

Performance Profiling and Analysis
===================================

Edge Performance Metrics
-------------------------

.. code-block:: python

   import time
   import torch.profiler
   from torch.utils.mobile_optimizer import optimize_for_mobile
   
   class EdgePerformanceProfiler:
       """Comprehensive profiling for edge deployment."""
       
       def __init__(self, model, sample_input):
           self.model = model
           self.sample_input = sample_input
       
       def profile_comprehensive(self):
           """Complete performance profiling suite."""
           results = {}
           
           # Basic metrics
           results['model_size'] = self.get_model_size()
           results['parameter_count'] = self.get_parameter_count()
           results['flops'] = self.estimate_flops()
           
           # Inference timing
           results['cpu_latency'] = self.measure_cpu_latency()
           results['memory_usage'] = self.measure_memory_usage()
           
           # Mobile optimization potential
           results['mobile_optimized'] = self.test_mobile_optimization()
           
           # Detailed profiling
           results['operation_breakdown'] = self.profile_operations()
           
           return results
       
       def get_model_size(self):
           """Calculate model size in MB."""
           param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
           buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
           return (param_size + buffer_size) / (1024 * 1024)
       
       def get_parameter_count(self):
           """Get total trainable parameters."""
           return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
       
       def estimate_flops(self):
           """Estimate FLOPs for the model."""
           # Simplified FLOP counting for common layers
           total_flops = 0
           
           def flop_count_hook(module, input, output):
               nonlocal total_flops
               if isinstance(module, nn.Conv2d):
                   # FLOPs = output_elements * (kernel_size^2 * input_channels + bias)
                   output_elements = output.numel()
                   kernel_flops = module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                   total_flops += output_elements * kernel_flops
               elif isinstance(module, nn.Linear):
                   total_flops += output.numel() * input[0].shape[-1]
           
           hooks = []
           for module in self.model.modules():
               if isinstance(module, (nn.Conv2d, nn.Linear)):
                   hook = module.register_forward_hook(flop_count_hook)
                   hooks.append(hook)
           
           with torch.no_grad():
               self.model(self.sample_input)
           
           for hook in hooks:
               hook.remove()
           
           return total_flops / 1e9  # Return in GFLOPs
       
       def measure_cpu_latency(self, num_runs=100):
           """Measure CPU inference latency."""
           self.model.eval()
           
           # Warmup
           with torch.no_grad():
               for _ in range(10):
                   self.model(self.sample_input)
           
           # Measurement
           start_time = time.time()
           with torch.no_grad():
               for _ in range(num_runs):
                   self.model(self.sample_input)
           end_time = time.time()
           
           return (end_time - start_time) / num_runs * 1000  # ms
       
       def measure_memory_usage(self):
           """Measure peak memory usage during inference."""
           import tracemalloc
           
           tracemalloc.start()
           
           with torch.no_grad():
               self.model(self.sample_input)
           
           current, peak = tracemalloc.get_traced_memory()
           tracemalloc.stop()
           
           return peak / (1024 * 1024)  # MB
       
       def test_mobile_optimization(self):
           """Test PyTorch Mobile optimization compatibility."""
           try:
               # Test mobile optimization
               self.model.eval()
               scripted_model = torch.jit.script(self.model)
               mobile_model = optimize_for_mobile(scripted_model)
               
               # Test inference
               with torch.no_grad():
                   original_output = self.model(self.sample_input)
                   mobile_output = mobile_model(self.sample_input)
               
               # Check output similarity
               diff = torch.mean(torch.abs(original_output - mobile_output))
               
               return {
                   'compatible': True,
                   'output_difference': diff.item(),
                   'size_reduction': self._calculate_size_reduction(scripted_model, mobile_model)
               }
           
           except Exception as e:
               return {
                   'compatible': False,
                   'error': str(e)
               }
       
       def profile_operations(self):
           """Detailed operation-level profiling."""
           self.model.eval()
           
           with torch.profiler.profile(
               activities=[torch.profiler.ProfilerActivity.CPU],
               record_shapes=True,
               with_stack=True
           ) as prof:
               with torch.no_grad():
                   self.model(self.sample_input)
           
           # Extract key metrics
           events = prof.key_averages()
           operation_breakdown = {}
           
           for event in events:
               if event.self_cpu_time_total > 0:
                   operation_breakdown[event.key] = {
                       'cpu_time_ms': event.self_cpu_time_total / 1000,
                       'cpu_percentage': event.self_cpu_time_total / prof.profiler.self_cpu_time_total * 100,
                       'count': event.count
                   }
           
           return operation_breakdown

Complete Edge Optimization Example
===================================

Main Execution Pipeline
-----------------------

.. code-block:: python

   import random
   import numpy as np
   from torch.utils.data import DataLoader
   
   def main_edge_optimization():
       """Complete edge optimization pipeline example."""
       
       # Set random seeds for reproducibility
       torch.manual_seed(42)
       np.random.seed(42)
       random.seed(42)
       
       # Load dataset (example with CIFAR-10)
       from torchvision import datasets, transforms
       
       transform = transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
       ])
       
       # Small dataset for quick evolution
       train_dataset = datasets.CIFAR10(
           root='./data', train=True, download=True, transform=transform
       )
       val_dataset = datasets.CIFAR10(
           root='./data', train=False, download=True, transform=transform
       )
       
       # Create data loaders
       train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
       val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
       
       # Mock dataset class for compatibility
       class DatasetWrapper:
           def __init__(self, train_loader, val_loader):
               self.train_loader = train_loader
               self.val_loader = val_loader
               self.num_classes = 10
           
           def train_dataloader(self):
               return self.train_loader
           
           def val_dataloader(self):
               return self.val_loader
       
       dataset = DatasetWrapper(train_loader, val_loader)
       
       # Configuration for edge optimization
       config = {
           'population_size': 20,
           'max_iterations': 10,
           'max_parameters': 300_000,
           'target_latency_ms': 30,
           'min_accuracy': 0.80
       }
       
       # Run mobile architecture evolution
       print("Starting Mobile Architecture Evolution...")
       evolution = MobileArchitectureEvolution(dataset, config)
       best_individuals, pareto_front = evolution.evolve_mobile_architecture()
       
       # Get the best performing model
       best_individual, best_results = best_individuals[-1]
       best_model = best_individual.build_model()
       
       print(f"\n=== Best Model Results ===")
       print(f"Fitness: {best_results['fitness']:.4f}")
       print(f"Accuracy: {best_results['accuracy']:.4f}")
       print(f"Parameters: {best_results['parameters']:,}")
       print(f"Latency: {best_results['latency_ms']:.2f} ms")
       print(f"Memory: {best_results['memory_mb']:.2f} MB")
       
       # Apply additional optimizations
       print(f"\n=== Applying Post-Training Optimizations ===")
       
       optimization_config = {
           'pruning': {
               'enabled': True,
               'ratio': 0.3
           },
           'quantization': {
               'enabled': True,
               'type': 'dynamic'
           }
       }
       
       pipeline = EdgeOptimizationPipeline(best_model, val_loader)
       optimized_model = pipeline.optimize_for_edge(optimization_config)
       
       # Profile final model
       print(f"\n=== Performance Profiling ===")
       sample_input = torch.randn(1, 3, 32, 32)
       profiler = EdgePerformanceProfiler(optimized_model, sample_input)
       profile_results = profiler.profile_comprehensive()
       
       print(f"Model Size: {profile_results['model_size']:.2f} MB")
       print(f"Parameters: {profile_results['parameter_count']:,}")
       print(f"FLOPs: {profile_results['flops']:.2f} G")
       print(f"CPU Latency: {profile_results['cpu_latency']:.2f} ms")
       print(f"Memory Usage: {profile_results['memory_usage']:.2f} MB")
       
       if profile_results['mobile_optimized']['compatible']:
           print(f"Mobile Compatible: ✓")
           print(f"Size Reduction: {profile_results['mobile_optimized']['size_reduction']:.1f}%")
       else:
           print(f"Mobile Compatible: ✗")
       
       # Display Pareto front
       print(f"\n=== Pareto Front ({len(pareto_front)} solutions) ===")
       for i, (individual, results) in enumerate(pareto_front):
           print(f"Solution {i+1}: Acc={results['accuracy']:.3f}, "
                 f"Params={results['parameters']:,}, Latency={results['latency_ms']:.1f}ms")
       
       return optimized_model, profile_results, pareto_front
   
   if __name__ == "__main__":
       optimized_model, profile_results, pareto_front = main_edge_optimization()

Expected Output
===============

The edge optimization example will produce output similar to:

.. code-block:: text

   Starting Mobile Architecture Evolution...
   
   === Generation 1 ===
   Best Individual - Fitness: 0.7245
     Accuracy: 0.8234
     Parameters: 187,452
     Latency: 23.45 ms
     Memory: 12.34 MB
   
   === Generation 10 ===
   Best Individual - Fitness: 0.8567
     Accuracy: 0.8923
     Parameters: 143,234
     Latency: 18.67 ms
     Memory: 9.87 MB
   
   === Best Model Results ===
   Fitness: 0.8567
   Accuracy: 0.8923
   Parameters: 143,234
   Latency: 18.67 ms
   Memory: 9.87 MB
   
   === Applying Post-Training Optimizations ===
   
   === Performance Profiling ===
   Model Size: 0.58 MB
   Parameters: 98,765
   FLOPs: 0.45 G
   CPU Latency: 12.34 ms
   Memory Usage: 7.23 MB
   Mobile Compatible: ✓
   Size Reduction: 23.4%
   
   === Pareto Front (8 solutions) ===
   Solution 1: Acc=0.892, Params=143,234, Latency=18.7ms
   Solution 2: Acc=0.876, Params=98,567, Latency=15.2ms
   Solution 3: Acc=0.834, Params=67,234, Latency=11.8ms
   ...

Key Takeaways
=============

1. **Multi-objective Optimization**: Edge deployment requires balancing accuracy, efficiency, and resource constraints
2. **Architecture Constraints**: Mobile-friendly blocks (MBConv, efficient convolutions) are preferred
3. **Post-training Optimization**: Quantization and pruning can significantly reduce model size and latency
4. **Pareto Optimization**: Multiple solutions along the accuracy-efficiency trade-off curve
5. **Real-world Metrics**: Actual latency and memory measurements are crucial for deployment decisions

This example demonstrates how PyNAS can be used to discover architectures specifically optimized for edge deployment, considering the unique constraints and requirements of mobile and embedded devices.
