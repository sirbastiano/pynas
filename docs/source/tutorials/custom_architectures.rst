Custom Architectures
===================

This tutorial covers how to create custom neural network architectures in PyNAS, including defining custom blocks, creating specialized architectures, and integrating them into the evolutionary process.

Overview
--------

PyNAS provides a flexible framework for defining custom architectures through:

- **Custom Blocks**: Define new building blocks for your architectures
- **Architecture Templates**: Create structured architecture patterns
- **Block Vocabulary**: Manage available blocks for evolution
- **Constraints**: Apply architectural constraints during evolution

Creating Custom Blocks
----------------------

Custom Convolution Block
~~~~~~~~~~~~~~~~~~~~~~~~

Here's how to create a custom convolution block with specific properties:

.. code-block:: python

    import torch
    import torch.nn as nn
    from pynas.blocks.convolutions import ConvBlock
    
    class DepthwiseSeparableConv(nn.Module):
        """
        Custom depthwise separable convolution block.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            kernel_size (int): Kernel size for convolution
            stride (int): Stride for convolution
            padding (int): Padding for convolution
        """
        
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            super().__init__()
            self.depthwise = nn.Conv2d(
                in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
            )
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x):
            x = self.relu(self.bn1(self.depthwise(x)))
            x = self.relu(self.bn2(self.pointwise(x)))
            return x

Custom Attention Block
~~~~~~~~~~~~~~~~~~~~~~

Create an attention mechanism for your architectures:

.. code-block:: python

    class SpatialAttention(nn.Module):
        """
        Spatial attention mechanism for feature enhancement.
        
        Args:
            channels (int): Number of input channels
            reduction (int): Channel reduction ratio
        """
        
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(channels, channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // reduction, channels, 1, bias=False)
            )
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            attention = self.sigmoid(avg_out + max_out)
            return x * attention

Integrating Custom Blocks
--------------------------

Register with Vocabulary
~~~~~~~~~~~~~~~~~~~~~~~~~

Add your custom blocks to the PyNAS vocabulary:

.. code-block:: python

    from pynas.core.vocabulary import Vocabulary
    
    # Create vocabulary instance
    vocab = Vocabulary()
    
    # Register custom blocks
    vocab.add_block('depthwise_separable', DepthwiseSeparableConv)
    vocab.add_block('spatial_attention', SpatialAttention)
    
    # Define block parameters
    vocab.set_block_params('depthwise_separable', {
        'kernel_size': [3, 5, 7],
        'stride': [1, 2]
    })
    
    vocab.set_block_params('spatial_attention', {
        'reduction': [8, 16, 32]
    })

Custom Architecture Template
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a template for generating architectures with your custom blocks:

.. code-block:: python

    from pynas.core.architecture_builder import ArchitectureBuilder
    
    class CustomArchitectureBuilder(ArchitectureBuilder):
        """
        Custom architecture builder with specific patterns.
        """
        
        def __init__(self, vocab, input_channels=3, num_classes=10):
            super().__init__(vocab)
            self.input_channels = input_channels
            self.num_classes = num_classes
        
        def create_encoder_block(self, in_ch, out_ch, block_type='depthwise_separable'):
            """Create encoder block with optional attention."""
            blocks = []
            
            # Add main convolution
            if block_type == 'depthwise_separable':
                blocks.append(DepthwiseSeparableConv(in_ch, out_ch))
            else:
                blocks.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
                blocks.append(nn.BatchNorm2d(out_ch))
                blocks.append(nn.ReLU(inplace=True))
            
            # Add attention if specified
            blocks.append(SpatialAttention(out_ch))
            
            return nn.Sequential(*blocks)
        
        def build_architecture(self, genome):
            """Build architecture from genome representation."""
            layers = []
            current_channels = self.input_channels
            
            for i, gene in enumerate(genome):
                block_type = gene['type']
                out_channels = gene['channels']
                
                layer = self.create_encoder_block(
                    current_channels, out_channels, block_type
                )
                layers.append(layer)
                current_channels = out_channels
            
            # Add final classifier
            layers.append(nn.AdaptiveAvgPool2d(1))
            layers.append(nn.Flatten())
            layers.append(nn.Linear(current_channels, self.num_classes))
            
            return nn.Sequential(*layers)

Evolutionary Configuration
--------------------------

Configure evolution with custom architectures:

.. code-block:: python

    from pynas.core.population import Population
    from pynas.core.individual import Individual
    from pynas.opt.evo import GeneticAlgorithm
    
    # Define custom genome structure
    def create_custom_genome():
        """Create genome for custom architecture."""
        genome = []
        channels = [32, 64, 128, 256]
        
        for ch in channels:
            gene = {
                'type': np.random.choice(['depthwise_separable', 'conv']),
                'channels': ch,
                'kernel_size': np.random.choice([3, 5, 7]),
                'use_attention': np.random.choice([True, False])
            }
            genome.append(gene)
        
        return genome
    
    # Initialize population with custom genomes
    population = Population(
        population_size=20,
        genome_factory=create_custom_genome,
        architecture_builder=CustomArchitectureBuilder(vocab)
    )
    
    # Configure genetic algorithm
    ga = GeneticAlgorithm(
        population=population,
        mutation_rate=0.1,
        crossover_rate=0.8,
        selection_method='tournament'
    )

Advanced Customization
----------------------

Multi-Scale Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

Create architectures that process multiple scales:

.. code-block:: python

    class MultiScaleBlock(nn.Module):
        """
        Multi-scale processing block with parallel paths.
        """
        
        def __init__(self, in_channels, out_channels):
            super().__init__()
            mid_channels = out_channels // 4
            
            self.scale1 = nn.Conv2d(in_channels, mid_channels, 1)
            self.scale2 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
            self.scale3 = nn.Conv2d(in_channels, mid_channels, 5, padding=2)
            self.scale4 = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_channels, mid_channels, 1)
            )
            
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
        
        def forward(self, x):
            s1 = self.scale1(x)
            s2 = self.scale2(x)
            s3 = self.scale3(x)
            s4 = self.scale4(x)
            
            out = torch.cat([s1, s2, s3, s4], dim=1)
            return self.relu(self.bn(out))

Conditional Architecture
~~~~~~~~~~~~~~~~~~~~~~~~

Create architectures that adapt based on conditions:

.. code-block:: python

    class ConditionalBlock(nn.Module):
        """
        Block that selects operation based on learned gating.
        """
        
        def __init__(self, in_channels, out_channels, num_ops=3):
            super().__init__()
            self.num_ops = num_ops
            
            # Define multiple operations
            self.ops = nn.ModuleList([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                DepthwiseSeparableConv(in_channels, out_channels),
                MultiScaleBlock(in_channels, out_channels)
            ])
            
            # Gating mechanism
            self.gating = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, num_ops),
                nn.Softmax(dim=1)
            )
        
        def forward(self, x):
            gates = self.gating(x).unsqueeze(-1).unsqueeze(-1)
            
            outputs = []
            for i, op in enumerate(self.ops):
                outputs.append(gates[:, i:i+1] * op(x))
            
            return sum(outputs)

Architecture Constraints
------------------------

Apply constraints during evolution:

.. code-block:: python

    class ArchitectureConstraints:
        """
        Define constraints for architecture evolution.
        """
        
        def __init__(self, max_params=1e6, max_flops=1e9):
            self.max_params = max_params
            self.max_flops = max_flops
        
        def check_constraints(self, individual):
            """Check if individual satisfies constraints."""
            model = individual.phenotype
            
            # Check parameter count
            param_count = sum(p.numel() for p in model.parameters())
            if param_count > self.max_params:
                return False
            
            # Check FLOPs (simplified estimation)
            flops = self.estimate_flops(model)
            if flops > self.max_flops:
                return False
            
            return True
        
        def estimate_flops(self, model):
            """Estimate FLOPs for the model."""
            # Simplified FLOP estimation
            total_flops = 0
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    # Rough FLOP estimation for conv layers
                    kernel_flops = module.kernel_size[0] * module.kernel_size[1]
                    output_elements = 224 * 224  # Assume input size
                    total_flops += kernel_flops * output_elements * module.in_channels * module.out_channels
            
            return total_flops

Complete Example
----------------

Here's a complete example putting it all together:

.. code-block:: python

    import torch
    import torch.nn as nn
    import numpy as np
    from pynas.core.population import Population
    from pynas.opt.evo import GeneticAlgorithm
    
    def run_custom_nas():
        """Run NAS with custom architectures."""
        
        # Setup vocabulary with custom blocks
        vocab = setup_custom_vocabulary()
        
        # Create custom architecture builder
        builder = CustomArchitectureBuilder(vocab, input_channels=3, num_classes=10)
        
        # Initialize population
        population = Population(
            population_size=20,
            genome_factory=create_custom_genome,
            architecture_builder=builder
        )
        
        # Setup constraints
        constraints = ArchitectureConstraints(max_params=1e6, max_flops=1e9)
        
        # Configure genetic algorithm
        ga = GeneticAlgorithm(
            population=population,
            mutation_rate=0.1,
            crossover_rate=0.8,
            constraints=constraints
        )
        
        # Run evolution
        for generation in range(50):
            # Evaluate population
            ga.evaluate_population()
            
            # Apply constraints
            ga.filter_by_constraints()
            
            # Evolve
            ga.evolve()
            
            # Log progress
            best_individual = ga.get_best_individual()
            print(f"Generation {generation}: Best fitness = {best_individual.fitness}")
        
        return ga.get_best_individual()
    
    if __name__ == "__main__":
        best_architecture = run_custom_nas()
        print(f"Best architecture found: {best_architecture}")

Next Steps
----------

After creating custom architectures:

1. **Validation**: Test your custom blocks thoroughly
2. **Integration**: Ensure compatibility with PyNAS framework
3. **Performance**: Profile your custom blocks for efficiency
4. **Documentation**: Document your custom components
5. **Sharing**: Consider contributing useful blocks back to PyNAS

.. seealso::
   
   - :doc:`advanced_config` for advanced configuration options
   - :doc:`../examples/custom_blocks` for more custom block examples
   - :doc:`../api/blocks` for API reference on blocks
