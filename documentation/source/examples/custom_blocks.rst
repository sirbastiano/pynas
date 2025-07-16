================
Custom Blocks
================

This example demonstrates how to create custom neural network blocks and integrate them into the PyNAS architecture search framework. You'll learn to extend the existing block vocabulary and implement domain-specific architectural components.

Overview
========

PyNAS provides a flexible framework for defining custom blocks that can be used in architecture search. This includes:

- **Custom Convolution Blocks**: Specialized convolutions for specific tasks
- **Attention Mechanisms**: Self-attention and cross-attention blocks
- **Domain-Specific Modules**: Blocks tailored for specific applications
- **Block Integration**: Adding custom blocks to the search space

.. contents:: Table of Contents
   :local:
   :depth: 2

Creating Custom Convolution Blocks
===================================

Basic Custom Block Structure
-----------------------------

.. code-block:: python

   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from pynas.blocks.activations import ReLU
   from pynas.blocks.convolutions import ConvBnAct
   
   class SeparableConvBlock(nn.Module):
       """
       Depthwise separable convolution block.
       
       This block implements depthwise separable convolution which is more
       parameter-efficient than standard convolution.
       
       Args:
           in_channels (int): Number of input channels
           out_channels (int): Number of output channels
           kernel_size (int): Kernel size for depthwise convolution
           stride (int): Stride for convolution
           padding (int): Padding for convolution
           activation (nn.Module): Activation function class
       """
       
       def __init__(self, in_channels, out_channels, kernel_size=3, 
                    stride=1, padding=1, activation=ReLU):
           super(SeparableConvBlock, self).__init__()
           
           # Depthwise convolution
           self.depthwise = nn.Conv2d(
               in_channels, in_channels, kernel_size=kernel_size,
               stride=stride, padding=padding, groups=in_channels, bias=False
           )
           self.bn1 = nn.BatchNorm2d(in_channels)
           
           # Pointwise convolution
           self.pointwise = nn.Conv2d(
               in_channels, out_channels, kernel_size=1, bias=False
           )
           self.bn2 = nn.BatchNorm2d(out_channels)
           
           self.activation = activation()
       
       def forward(self, x):
           """Forward pass through separable convolution."""
           # Depthwise
           x = self.depthwise(x)
           x = self.bn1(x)
           x = self.activation(x)
           
           # Pointwise
           x = self.pointwise(x)
           x = self.bn2(x)
           x = self.activation(x)
           
           return x
   
   class DilatedConvBlock(nn.Module):
       """
       Dilated convolution block for capturing multi-scale features.
       
       Args:
           in_channels (int): Number of input channels
           out_channels (int): Number of output channels
           dilation_rates (list): List of dilation rates to use
           activation (nn.Module): Activation function class
       """
       
       def __init__(self, in_channels, out_channels, 
                    dilation_rates=[1, 2, 4], activation=ReLU):
           super(DilatedConvBlock, self).__init__()
           
           self.dilated_convs = nn.ModuleList()
           
           for dilation in dilation_rates:
               conv = nn.Sequential(
                   nn.Conv2d(
                       in_channels, out_channels // len(dilation_rates),
                       kernel_size=3, padding=dilation, dilation=dilation, bias=False
                   ),
                   nn.BatchNorm2d(out_channels // len(dilation_rates)),
                   activation()
               )
               self.dilated_convs.append(conv)
           
           # Final 1x1 conv to combine features
           self.combine = nn.Sequential(
               nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
               nn.BatchNorm2d(out_channels),
               activation()
           )
       
       def forward(self, x):
           """Forward pass through dilated convolution block."""
           # Apply different dilated convolutions
           features = []
           for conv in self.dilated_convs:
               features.append(conv(x))
           
           # Concatenate features
           combined = torch.cat(features, dim=1)
           
           # Final combination
           output = self.combine(combined)
           
           return output

Attention Mechanism Blocks
===========================

Self-Attention Block
--------------------

.. code-block:: python

   import math
   
   class SelfAttentionBlock(nn.Module):
       """
       Self-attention block for capturing long-range dependencies.
       
       Args:
           in_channels (int): Number of input channels
           reduction_ratio (int): Reduction ratio for attention computation
           activation (nn.Module): Activation function class
       """
       
       def __init__(self, in_channels, reduction_ratio=8, activation=ReLU):
           super(SelfAttentionBlock, self).__init__()
           
           self.in_channels = in_channels
           self.reduction_ratio = reduction_ratio
           
           # Query, Key, Value projections
           self.query_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
           self.key_conv = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
           self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
           
           # Output projection
           self.output_conv = nn.Conv2d(in_channels, in_channels, 1)
           
           # Layer normalization
           self.layer_norm = nn.GroupNorm(1, in_channels)
           
           self.activation = activation()
           self.softmax = nn.Softmax(dim=-1)
           
       def forward(self, x):
           """Forward pass through self-attention block."""
           batch_size, channels, height, width = x.size()
           
           # Store residual
           residual = x
           
           # Generate queries, keys, values
           queries = self.query_conv(x).view(batch_size, -1, height * width)
           keys = self.key_conv(x).view(batch_size, -1, height * width)
           values = self.value_conv(x).view(batch_size, -1, height * width)
           
           # Compute attention weights
           attention_weights = torch.bmm(queries.transpose(1, 2), keys)
           attention_weights = attention_weights / math.sqrt(channels // self.reduction_ratio)
           attention_weights = self.softmax(attention_weights)
           
           # Apply attention to values
           attended_values = torch.bmm(values, attention_weights.transpose(1, 2))
           attended_values = attended_values.view(batch_size, channels, height, width)
           
           # Output projection
           output = self.output_conv(attended_values)
           
           # Residual connection and layer norm
           output = residual + output
           output = self.layer_norm(output)
           
           return output
   
   class ChannelAttentionBlock(nn.Module):
       """
       Channel attention block (enhanced Squeeze-and-Excitation).
       
       Args:
           in_channels (int): Number of input channels
           reduction_ratio (int): Reduction ratio for channel attention
           use_spatial (bool): Whether to include spatial attention
           activation (nn.Module): Activation function class
       """
       
       def __init__(self, in_channels, reduction_ratio=16, 
                    use_spatial=True, activation=ReLU):
           super(ChannelAttentionBlock, self).__init__()
           
           self.in_channels = in_channels
           self.use_spatial = use_spatial
           
           # Channel attention
           self.avg_pool = nn.AdaptiveAvgPool2d(1)
           self.max_pool = nn.AdaptiveMaxPool2d(1)
           
           reduced_channels = max(in_channels // reduction_ratio, 1)
           
           self.channel_attention = nn.Sequential(
               nn.Linear(in_channels, reduced_channels, bias=False),
               activation(),
               nn.Linear(reduced_channels, in_channels, bias=False)
           )
           
           # Spatial attention (optional)
           if use_spatial:
               self.spatial_attention = nn.Sequential(
                   nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                   nn.Sigmoid()
               )
           
           self.sigmoid = nn.Sigmoid()
       
       def forward(self, x):
           """Forward pass through channel attention block."""
           batch_size, channels, height, width = x.size()
           
           # Channel attention
           avg_pooled = self.avg_pool(x).view(batch_size, channels)
           max_pooled = self.max_pool(x).view(batch_size, channels)
           
           avg_attention = self.channel_attention(avg_pooled)
           max_attention = self.channel_attention(max_pooled)
           
           channel_attention = self.sigmoid(avg_attention + max_attention)
           channel_attention = channel_attention.view(batch_size, channels, 1, 1)
           
           # Apply channel attention
           x = x * channel_attention
           
           # Spatial attention (if enabled)
           if self.use_spatial:
               avg_spatial = torch.mean(x, dim=1, keepdim=True)
               max_spatial, _ = torch.max(x, dim=1, keepdim=True)
               spatial_input = torch.cat([avg_spatial, max_spatial], dim=1)
               
               spatial_attention = self.spatial_attention(spatial_input)
               x = x * spatial_attention
           
           return x

Domain-Specific Blocks
=======================

Remote Sensing Block
--------------------

.. code-block:: python

   class RemoteSensingBlock(nn.Module):
       """
       Specialized block for remote sensing applications.
       
       This block is designed to handle multi-spectral imagery and
       capture both spectral and spatial features effectively.
       
       Args:
           in_channels (int): Number of input channels (spectral bands)
           out_channels (int): Number of output channels
           spectral_reduction (int): Reduction factor for spectral processing
           activation (nn.Module): Activation function class
       """
       
       def __init__(self, in_channels, out_channels, 
                    spectral_reduction=4, activation=ReLU):
           super(RemoteSensingBlock, self).__init__()
           
           self.in_channels = in_channels
           self.out_channels = out_channels
           
           # Spectral feature extraction
           self.spectral_conv = nn.Sequential(
               nn.Conv2d(in_channels, in_channels // spectral_reduction, 1),
               nn.BatchNorm2d(in_channels // spectral_reduction),
               activation(),
               nn.Conv2d(in_channels // spectral_reduction, out_channels // 2, 1),
               nn.BatchNorm2d(out_channels // 2),
               activation()
           )
           
           # Spatial feature extraction with different scales
           self.spatial_conv_3x3 = nn.Sequential(
               nn.Conv2d(in_channels, out_channels // 4, 3, padding=1),
               nn.BatchNorm2d(out_channels // 4),
               activation()
           )
           
           self.spatial_conv_5x5 = nn.Sequential(
               nn.Conv2d(in_channels, out_channels // 4, 5, padding=2),
               nn.BatchNorm2d(out_channels // 4),
               activation()
           )
           
           # Feature fusion
           self.fusion = nn.Sequential(
               nn.Conv2d(out_channels, out_channels, 1),
               nn.BatchNorm2d(out_channels),
               activation()
           )
           
           # Adaptive feature weighting
           self.feature_weighting = nn.Sequential(
               nn.AdaptiveAvgPool2d(1),
               nn.Conv2d(out_channels, out_channels // 4, 1),
               activation(),
               nn.Conv2d(out_channels // 4, out_channels, 1),
               nn.Sigmoid()
           )
       
       def forward(self, x):
           """Forward pass through remote sensing block."""
           # Extract different types of features
           spectral_features = self.spectral_conv(x)
           spatial_features_3x3 = self.spatial_conv_3x3(x)
           spatial_features_5x5 = self.spatial_conv_5x5(x)
           
           # Combine features
           combined_features = torch.cat([
               spectral_features, 
               spatial_features_3x3, 
               spatial_features_5x5
           ], dim=1)
           
           # Feature fusion
           fused_features = self.fusion(combined_features)
           
           # Adaptive weighting
           weights = self.feature_weighting(fused_features)
           output = fused_features * weights
           
           return output
   
   class EdgeDetectionBlock(nn.Module):
       """
       Edge detection block using learned filters.
       
       Args:
           in_channels (int): Number of input channels
           out_channels (int): Number of output channels
           edge_types (list): Types of edges to detect ('horizontal', 'vertical', 'diagonal')
           activation (nn.Module): Activation function class
       """
       
       def __init__(self, in_channels, out_channels, 
                    edge_types=['horizontal', 'vertical', 'diagonal'], 
                    activation=ReLU):
           super(EdgeDetectionBlock, self).__init__()
           
           self.edge_types = edge_types
           self.edge_detectors = nn.ModuleList()
           
           # Create edge detection filters for each type
           for edge_type in edge_types:
               if edge_type == 'horizontal':
                   # Horizontal edge detection
                   detector = nn.Conv2d(
                       in_channels, out_channels // len(edge_types), 
                       kernel_size=3, padding=1, bias=False
                   )
                   # Initialize with horizontal edge filter
                   with torch.no_grad():
                       detector.weight.fill_(0)
                       detector.weight[:, :, 0, :] = -1
                       detector.weight[:, :, 2, :] = 1
               
               elif edge_type == 'vertical':
                   # Vertical edge detection
                   detector = nn.Conv2d(
                       in_channels, out_channels // len(edge_types), 
                       kernel_size=3, padding=1, bias=False
                   )
                   with torch.no_grad():
                       detector.weight.fill_(0)
                       detector.weight[:, :, :, 0] = -1
                       detector.weight[:, :, :, 2] = 1
               
               elif edge_type == 'diagonal':
                   # Diagonal edge detection
                   detector = nn.Conv2d(
                       in_channels, out_channels // len(edge_types), 
                       kernel_size=3, padding=1, bias=False
                   )
                   with torch.no_grad():
                       detector.weight.fill_(0)
                       detector.weight[:, :, 0, 0] = -1
                       detector.weight[:, :, 2, 2] = 1
               
               self.edge_detectors.append(nn.Sequential(
                   detector,
                   nn.BatchNorm2d(out_channels // len(edge_types)),
                   activation()
               ))
           
           # Feature combination
           self.combiner = nn.Sequential(
               nn.Conv2d(out_channels, out_channels, 1),
               nn.BatchNorm2d(out_channels),
               activation()
           )
       
       def forward(self, x):
           """Forward pass through edge detection block."""
           edge_features = []
           
           for detector in self.edge_detectors:
               edge_feature = detector(x)
               edge_features.append(edge_feature)
           
           # Combine edge features
           combined = torch.cat(edge_features, dim=1)
           output = self.combiner(combined)
           
           return output

Integrating Custom Blocks into PyNAS
=====================================

Extending the Block Vocabulary
-------------------------------

.. code-block:: python

   # File: custom_vocabulary.py
   
   from pynas.core.vocabulary import (
       convolution_layer_vocabulary,
       layer_parameters
   )
   
   # Extend the vocabulary with custom blocks
   custom_convolution_vocabulary = {
       **convolution_layer_vocabulary,
       'sep': 'SeparableConvBlock',
       'dil': 'DilatedConvBlock', 
       'sa': 'SelfAttentionBlock',
       'ca': 'ChannelAttentionBlock',
       'rs': 'RemoteSensingBlock',
       'edge': 'EdgeDetectionBlock'
   }
   
   # Add parameters for custom blocks
   custom_layer_parameters = {
       **layer_parameters,
       'SeparableConvBlock': [
           'out_channels_coefficient', 'kernel_size', 'stride', 'padding', 'activation'
       ],
       'DilatedConvBlock': [
           'out_channels_coefficient', 'dilation_rates', 'activation'
       ],
       'SelfAttentionBlock': [
           'reduction_ratio', 'activation'
       ],
       'ChannelAttentionBlock': [
           'reduction_ratio', 'use_spatial', 'activation'
       ],
       'RemoteSensingBlock': [
           'out_channels_coefficient', 'spectral_reduction', 'activation'
       ],
       'EdgeDetectionBlock': [
           'out_channels_coefficient', 'edge_types', 'activation'
       ]
   }

Custom Block Registry
---------------------

.. code-block:: python

   # File: custom_blocks_registry.py
   
   import importlib
   from typing import Dict, Any, Type
   import torch.nn as nn
   
   class CustomBlockRegistry:
       """Registry for managing custom blocks in PyNAS."""
       
       def __init__(self):
           self.blocks = {}
           self.parameter_configs = {}
           
           # Register built-in custom blocks
           self._register_builtin_blocks()
       
       def register_block(self, name: str, block_class: Type[nn.Module], 
                         parameters: list, default_config: Dict[str, Any] = None):
           """
           Register a custom block.
           
           Args:
               name: Name identifier for the block
               block_class: PyTorch Module class
               parameters: List of parameter names for the block
               default_config: Default parameter values
           """
           self.blocks[name] = block_class
           self.parameter_configs[name] = {
               'parameters': parameters,
               'defaults': default_config or {}
           }
           
           print(f"Registered custom block: {name}")
       
       def get_block(self, name: str) -> Type[nn.Module]:
           """Get block class by name."""
           if name not in self.blocks:
               raise ValueError(f"Block '{name}' not found in registry")
           return self.blocks[name]
       
       def get_parameters(self, name: str) -> list:
           """Get parameter list for a block."""
           if name not in self.parameter_configs:
               raise ValueError(f"Block '{name}' not found in registry")
           return self.parameter_configs[name]['parameters']
       
       def get_defaults(self, name: str) -> Dict[str, Any]:
           """Get default configuration for a block."""
           if name not in self.parameter_configs:
               raise ValueError(f"Block '{name}' not found in registry")
           return self.parameter_configs[name]['defaults']
       
       def list_blocks(self) -> list:
           """List all registered blocks."""
           return list(self.blocks.keys())
       
       def _register_builtin_blocks(self):
           """Register built-in custom blocks."""
           # Register the custom blocks we defined
           self.register_block(
               'SeparableConvBlock', 
               SeparableConvBlock,
               ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'activation'],
               {'kernel_size': 3, 'stride': 1, 'padding': 1}
           )
           
           self.register_block(
               'DilatedConvBlock',
               DilatedConvBlock, 
               ['in_channels', 'out_channels', 'dilation_rates', 'activation'],
               {'dilation_rates': [1, 2, 4]}
           )
           
           self.register_block(
               'SelfAttentionBlock',
               SelfAttentionBlock,
               ['in_channels', 'reduction_ratio', 'activation'],
               {'reduction_ratio': 8}
           )
           
           self.register_block(
               'ChannelAttentionBlock',
               ChannelAttentionBlock,
               ['in_channels', 'reduction_ratio', 'use_spatial', 'activation'],
               {'reduction_ratio': 16, 'use_spatial': True}
           )
           
           self.register_block(
               'RemoteSensingBlock',
               RemoteSensingBlock,
               ['in_channels', 'out_channels', 'spectral_reduction', 'activation'],
               {'spectral_reduction': 4}
           )
           
           self.register_block(
               'EdgeDetectionBlock',
               EdgeDetectionBlock,
               ['in_channels', 'out_channels', 'edge_types', 'activation'],
               {'edge_types': ['horizontal', 'vertical', 'diagonal']}
           )
   
   # Global registry instance
   custom_block_registry = CustomBlockRegistry()

Custom Architecture Builder
----------------------------

.. code-block:: python

   # File: custom_architecture_builder.py
   
   from pynas.core.architecture_builder import ArchitectureBuilder
   from pynas.core.generic_unet import build_layer, parse_conv_params
   import configparser
   
   class CustomArchitectureBuilder(ArchitectureBuilder):
       """Extended architecture builder with custom block support."""
       
       def __init__(self, custom_registry=None):
           super().__init__()
           self.custom_registry = custom_registry or custom_block_registry
           
           # Load custom configuration
           self.custom_config = self._load_custom_config()
       
       def _load_custom_config(self):
           """Load configuration for custom blocks."""
           config = configparser.ConfigParser()
           
           # Add configurations for custom blocks
           config.add_section('SeparableConvBlock')
           config.set('SeparableConvBlock', 'min_kernel_size', '3')
           config.set('SeparableConvBlock', 'max_kernel_size', '5')
           config.set('SeparableConvBlock', 'default_kernel_size', '3')
           config.set('SeparableConvBlock', 'min_out_channels_coefficient', '4')
           config.set('SeparableConvBlock', 'max_out_channels_coefficient', '12')
           config.set('SeparableConvBlock', 'default_out_channels_coefficient', '8')
           
           config.add_section('DilatedConvBlock')
           config.set('DilatedConvBlock', 'min_out_channels_coefficient', '4')
           config.set('DilatedConvBlock', 'max_out_channels_coefficient', '12')
           config.set('DilatedConvBlock', 'default_out_channels_coefficient', '8')
           
           config.add_section('SelfAttentionBlock')
           config.set('SelfAttentionBlock', 'min_reduction_ratio', '4')
           config.set('SelfAttentionBlock', 'max_reduction_ratio', '16')
           config.set('SelfAttentionBlock', 'default_reduction_ratio', '8')
           
           config.add_section('ChannelAttentionBlock')
           config.set('ChannelAttentionBlock', 'min_reduction_ratio', '8')
           config.set('ChannelAttentionBlock', 'max_reduction_ratio', '32')
           config.set('ChannelAttentionBlock', 'default_reduction_ratio', '16')
           
           config.add_section('RemoteSensingBlock')
           config.set('RemoteSensingBlock', 'min_out_channels_coefficient', '4')
           config.set('RemoteSensingBlock', 'max_out_channels_coefficient', '12')
           config.set('RemoteSensingBlock', 'default_out_channels_coefficient', '8')
           config.set('RemoteSensingBlock', 'min_spectral_reduction', '2')
           config.set('RemoteSensingBlock', 'max_spectral_reduction', '8')
           config.set('RemoteSensingBlock', 'default_spectral_reduction', '4')
           
           config.add_section('EdgeDetectionBlock')
           config.set('EdgeDetectionBlock', 'min_out_channels_coefficient', '4')
           config.set('EdgeDetectionBlock', 'max_out_channels_coefficient', '12')
           config.set('EdgeDetectionBlock', 'default_out_channels_coefficient', '8')
           
           return config
       
       def build_custom_layer(self, layer_config, current_channels, 
                             current_height, current_width, get_activation_fn):
           """Build a custom layer from configuration."""
           layer_type = layer_config['layer_type']
           
           if layer_type not in self.custom_registry.blocks:
               # Fall back to default builder
               return build_layer(
                   layer_config, self.custom_config, current_channels,
                   current_height, current_width, 0, get_activation_fn
               )
           
           # Get block class and parameters
           block_class = self.custom_registry.get_block(layer_type)
           
           # Parse parameters specific to this block type
           params = self._parse_custom_params(layer_config, layer_type, current_channels)
           
           # Create the layer instance
           layer_instance = block_class(**params)
           
           # Calculate output channels and dimensions
           out_channels = params.get('out_channels', current_channels)
           
           return layer_instance, out_channels, current_height, current_width
       
       def _parse_custom_params(self, layer_config, layer_type, current_channels):
           """Parse parameters for custom blocks."""
           params = {}
           defaults = self.custom_registry.get_defaults(layer_type)
           
           # Common parameters
           if 'out_channels_coefficient' in layer_config:
               coeff = layer_config['out_channels_coefficient']
               params['out_channels'] = int(current_channels * coeff)
           elif layer_type in self.custom_config:
               default_coeff = self.custom_config.getfloat(
                   layer_type, 'default_out_channels_coefficient'
               )
               params['out_channels'] = int(current_channels * default_coeff)
           
           params['in_channels'] = current_channels
           
           # Block-specific parameters
           if layer_type == 'SeparableConvBlock':
               params['kernel_size'] = layer_config.get('kernel_size', 3)
               params['stride'] = layer_config.get('stride', 1)
               params['padding'] = layer_config.get('padding', 1)
           
           elif layer_type == 'DilatedConvBlock':
               params['dilation_rates'] = layer_config.get(
                   'dilation_rates', defaults['dilation_rates']
               )
           
           elif layer_type == 'SelfAttentionBlock':
               params['reduction_ratio'] = layer_config.get('reduction_ratio', 8)
               # Remove out_channels for attention blocks (they preserve channels)
               params['out_channels'] = current_channels
           
           elif layer_type == 'ChannelAttentionBlock':
               params['reduction_ratio'] = layer_config.get('reduction_ratio', 16)
               params['use_spatial'] = layer_config.get('use_spatial', True)
               params['out_channels'] = current_channels
           
           elif layer_type == 'RemoteSensingBlock':
               params['spectral_reduction'] = layer_config.get('spectral_reduction', 4)
           
           elif layer_type == 'EdgeDetectionBlock':
               params['edge_types'] = layer_config.get(
                   'edge_types', defaults['edge_types']
               )
           
           # Activation function
           if 'activation' in layer_config:
               from pynas.blocks import activations
               activation_name = layer_config['activation']
               params['activation'] = getattr(activations, activation_name)
           
           return params

Evolution with Custom Blocks
=============================

Custom Block Evolution Example
-------------------------------

.. code-block:: python

   # File: custom_block_evolution.py
   
   import random
   import torch
   from pynas.core.population import Population
   from pynas.core.individual import Individual
   from custom_blocks_registry import custom_block_registry
   from custom_architecture_builder import CustomArchitectureBuilder
   
   class CustomBlockEvolution:
       """Evolution specifically using custom blocks."""
       
       def __init__(self, dataset, config):
           self.dataset = dataset
           self.config = config
           self.custom_builder = CustomArchitectureBuilder()
           
           # Define custom architecture templates
           self.custom_templates = [
               "sep2r_ca_sep2r_C",          # Separable + Channel Attention
               "rs3g_sa_dil2r_C",           # Remote Sensing + Self Attention + Dilated
               "edge1r_sep2r_ca_C",         # Edge Detection + Separable + Channel Attention
               "dil2g_sa_sep2r_C",          # Dilated + Self Attention + Separable
               "rs2r_edge1r_ca_sep1r_C",    # Complex multi-block architecture
           ]
       
       def create_custom_population(self):
           """Create population using custom blocks."""
           population = []
           
           for template in self.custom_templates:
               for variation in range(self.config['population_size'] // len(self.custom_templates)):
                   individual = self._create_custom_individual(template, variation)
                   population.append(individual)
           
           # Fill remaining spots with random custom architectures
           while len(population) < self.config['population_size']:
               random_individual = self._create_random_custom_individual()
               population.append(random_individual)
           
           return Population(individuals=population, config=self.config)
       
       def _create_custom_individual(self, template, variation):
           """Create individual from custom template with variations."""
           # Parse template into layers
           parsed_arch = self._parse_custom_template(template)
           
           # Apply variations
           if variation > 0:
               parsed_arch = self._apply_variations(parsed_arch, variation)
           
           return Individual(
               genome=parsed_arch,
               task=self.config.get('task', 'classification'),
               input_shape=self.config.get('input_shape', (3, 224, 224)),
               num_classes=self.dataset.num_classes
           )
       
       def _parse_custom_template(self, template):
           """Parse custom template string into architecture."""
           # Extended parsing for custom blocks
           custom_vocab = {
               'sep': 'SeparableConvBlock',
               'dil': 'DilatedConvBlock',
               'sa': 'SelfAttentionBlock', 
               'ca': 'ChannelAttentionBlock',
               'rs': 'RemoteSensingBlock',
               'edge': 'EdgeDetectionBlock',
               'r': 'ReLU',
               'g': 'GELU',
               'C': 'Classifier'
           }
           
           parsed_layers = []
           i = 0
           
           while i < len(template):
               if template[i:i+4] == 'edge':
                   # Edge detection block
                   i += 4
                   count = int(template[i]) if i < len(template) and template[i].isdigit() else 1
                   i += 1
                   activation = template[i] if i < len(template) else 'r'
                   i += 1
                   
                   for _ in range(count):
                       parsed_layers.append({
                           'layer_type': 'EdgeDetectionBlock',
                           'activation': 'ReLU' if activation == 'r' else 'GELU',
                           'out_channels_coefficient': 8,
                           'edge_types': ['horizontal', 'vertical', 'diagonal']
                       })
               
               elif template[i:i+3] == 'sep':
                   # Separable convolution block
                   i += 3
                   count = int(template[i]) if i < len(template) and template[i].isdigit() else 1
                   i += 1
                   activation = template[i] if i < len(template) else 'r'
                   i += 1
                   
                   for _ in range(count):
                       parsed_layers.append({
                           'layer_type': 'SeparableConvBlock',
                           'activation': 'ReLU' if activation == 'r' else 'GELU',
                           'out_channels_coefficient': 8,
                           'kernel_size': 3
                       })
               
               elif template[i:i+3] == 'dil':
                   # Dilated convolution block
                   i += 3
                   count = int(template[i]) if i < len(template) and template[i].isdigit() else 1
                   i += 1
                   activation = template[i] if i < len(template) else 'r'
                   i += 1
                   
                   for _ in range(count):
                       parsed_layers.append({
                           'layer_type': 'DilatedConvBlock',
                           'activation': 'ReLU' if activation == 'r' else 'GELU',
                           'out_channels_coefficient': 8,
                           'dilation_rates': [1, 2, 4]
                       })
               
               elif template[i:i+2] == 'rs':
                   # Remote sensing block
                   i += 2
                   count = int(template[i]) if i < len(template) and template[i].isdigit() else 1
                   i += 1
                   activation = template[i] if i < len(template) else 'r'
                   i += 1
                   
                   for _ in range(count):
                       parsed_layers.append({
                           'layer_type': 'RemoteSensingBlock',
                           'activation': 'ReLU' if activation == 'r' else 'GELU',
                           'out_channels_coefficient': 8,
                           'spectral_reduction': 4
                       })
               
               elif template[i:i+2] == 'sa':
                   # Self attention block
                   i += 2
                   parsed_layers.append({
                       'layer_type': 'SelfAttentionBlock',
                       'activation': 'ReLU',
                       'reduction_ratio': 8
                   })
               
               elif template[i:i+2] == 'ca':
                   # Channel attention block
                   i += 2
                   parsed_layers.append({
                       'layer_type': 'ChannelAttentionBlock',
                       'activation': 'ReLU',
                       'reduction_ratio': 16,
                       'use_spatial': True
                   })
               
               elif template[i] == '_':
                   # Separator
                   i += 1
               
               elif template[i] == 'C':
                   # Classifier head
                   parsed_layers.append({
                       'layer_type': 'Classifier',
                       'activation': 'ReLU'
                   })
                   i += 1
               
               else:
                   i += 1
           
           return parsed_layers
       
       def _apply_variations(self, parsed_arch, variation):
           """Apply variations to the base architecture."""
           varied_arch = parsed_arch.copy()
           
           for i, layer in enumerate(varied_arch):
               if variation == 1:
                   # Increase channel coefficients
                   if 'out_channels_coefficient' in layer:
                       layer['out_channels_coefficient'] = min(
                           layer['out_channels_coefficient'] + 2, 12
                       )
               
               elif variation == 2:
                   # Change activation functions
                   if layer.get('activation') == 'ReLU':
                       layer['activation'] = 'GELU'
               
               elif variation == 3:
                   # Modify attention parameters
                   if layer.get('layer_type') == 'SelfAttentionBlock':
                       layer['reduction_ratio'] = 4
                   elif layer.get('layer_type') == 'ChannelAttentionBlock':
                       layer['reduction_ratio'] = 8
           
           return varied_arch
       
       def _create_random_custom_individual(self):
           """Create random architecture using custom blocks."""
           custom_blocks = [
               'SeparableConvBlock', 'DilatedConvBlock', 'SelfAttentionBlock',
               'ChannelAttentionBlock', 'RemoteSensingBlock', 'EdgeDetectionBlock'
           ]
           
           num_layers = random.randint(3, 6)
           parsed_arch = []
           
           for _ in range(num_layers):
               block_type = random.choice(custom_blocks)
               
               layer = {
                   'layer_type': block_type,
                   'activation': random.choice(['ReLU', 'GELU']),
               }
               
               if block_type in ['SeparableConvBlock', 'DilatedConvBlock', 
                               'RemoteSensingBlock', 'EdgeDetectionBlock']:
                   layer['out_channels_coefficient'] = random.randint(6, 10)
               
               if block_type == 'SeparableConvBlock':
                   layer['kernel_size'] = random.choice([3, 5])
               
               elif block_type == 'SelfAttentionBlock':
                   layer['reduction_ratio'] = random.choice([4, 8, 16])
               
               elif block_type == 'ChannelAttentionBlock':
                   layer['reduction_ratio'] = random.choice([8, 16, 32])
                   layer['use_spatial'] = random.choice([True, False])
               
               elif block_type == 'RemoteSensingBlock':
                   layer['spectral_reduction'] = random.choice([2, 4, 8])
               
               parsed_arch.append(layer)
           
           # Add classifier head
           parsed_arch.append({
               'layer_type': 'Classifier',
               'activation': 'ReLU'
           })
           
           return Individual(
               genome=parsed_arch,
               task=self.config.get('task', 'classification'),
               input_shape=self.config.get('input_shape', (3, 224, 224)),
               num_classes=self.dataset.num_classes
           )
       
       def evolve_with_custom_blocks(self):
           """Run evolution using custom blocks."""
           population = self.create_custom_population()
           
           print(f"Starting evolution with {len(custom_block_registry.list_blocks())} custom blocks")
           print(f"Available blocks: {custom_block_registry.list_blocks()}")
           
           best_individuals = []
           
           for generation in range(self.config['max_iterations']):
               print(f"\n=== Generation {generation + 1} ===")
               
               # Evaluate population
               for individual in population.individuals:
                   try:
                       model = individual.build_model()
                       if model is not None:
                           # Simple fitness evaluation (accuracy-based)
                           fitness = self._evaluate_model(model)
                           individual.fitness = fitness
                       else:
                           individual.fitness = 0.0
                   except Exception as e:
                       print(f"Error building model: {e}")
                       individual.fitness = 0.0
               
               # Track best individual
               best_idx = max(range(len(population.individuals)), 
                            key=lambda i: population.individuals[i].fitness)
               best_individual = population.individuals[best_idx]
               
               print(f"Best Individual Fitness: {best_individual.fitness:.4f}")
               print(f"Architecture: {self._format_architecture(best_individual.genome)}")
               
               best_individuals.append(best_individual.copy())
               
               # Evolution step
               if generation < self.config['max_iterations'] - 1:
                   population = self._custom_evolution_step(population)
           
           return best_individuals
       
       def _evaluate_model(self, model):
           """Simple model evaluation (placeholder)."""
           # In a real scenario, this would evaluate on validation data
           # For this example, we'll use a simple heuristic
           param_count = sum(p.numel() for p in model.parameters())
           
           # Reward smaller models with reasonable complexity
           if param_count < 100000:
               return 0.5 + random.random() * 0.3
           elif param_count < 500000:
               return 0.6 + random.random() * 0.3
           else:
               return 0.4 + random.random() * 0.2
       
       def _format_architecture(self, genome):
           """Format architecture for display."""
           formatted = []
           for layer in genome:
               layer_type = layer.get('layer_type', 'Unknown')
               if layer_type == 'Classifier':
                   formatted.append('C')
               else:
                   formatted.append(layer_type[:3])
           return ' -> '.join(formatted)
       
       def _custom_evolution_step(self, population):
           """Evolution step preserving custom block characteristics."""
           # Selection (top 50%)
           population.individuals.sort(key=lambda x: x.fitness, reverse=True)
           selected = population.individuals[:len(population.individuals)//2]
           
           # Generate offspring
           offspring = []
           while len(offspring) < self.config['population_size']:
               parent1 = random.choice(selected)
               parent2 = random.choice(selected)
               
               child = self._custom_crossover(parent1, parent2)
               child = self._custom_mutation(child)
               
               offspring.append(child)
           
           return Population(individuals=offspring, config=self.config)
       
       def _custom_crossover(self, parent1, parent2):
           """Crossover preserving custom block structures."""
           # Simple single-point crossover
           p1_genome = parent1.genome[:-1]  # Exclude classifier
           p2_genome = parent2.genome[:-1]
           
           if len(p1_genome) > 0 and len(p2_genome) > 0:
               crossover_point = random.randint(1, min(len(p1_genome), len(p2_genome)))
               child_genome = p1_genome[:crossover_point] + p2_genome[crossover_point:]
           else:
               child_genome = p1_genome if len(p1_genome) > 0 else p2_genome
           
           # Add classifier
           child_genome.append({
               'layer_type': 'Classifier',
               'activation': 'ReLU'
           })
           
           return Individual(
               genome=child_genome,
               task=parent1.task,
               input_shape=parent1.input_shape,
               num_classes=parent1.num_classes
           )
       
       def _custom_mutation(self, individual):
           """Mutation respecting custom block constraints."""
           if random.random() < 0.3:  # 30% mutation rate
               genome = individual.genome[:-1]  # Exclude classifier
               
               if len(genome) > 0:
                   # Choose mutation type
                   mutation_type = random.choice(['modify', 'add', 'remove'])
                   
                   if mutation_type == 'modify':
                       # Modify existing layer
                       idx = random.randint(0, len(genome) - 1)
                       layer = genome[idx].copy()
                       
                       if 'out_channels_coefficient' in layer:
                           layer['out_channels_coefficient'] = random.randint(6, 12)
                       
                       if layer.get('layer_type') == 'SelfAttentionBlock':
                           layer['reduction_ratio'] = random.choice([4, 8, 16])
                       
                       genome[idx] = layer
                   
                   elif mutation_type == 'add' and len(genome) < 6:
                       # Add new custom block
                       new_block_type = random.choice([
                           'SeparableConvBlock', 'ChannelAttentionBlock', 'DilatedConvBlock'
                       ])
                       
                       new_layer = {
                           'layer_type': new_block_type,
                           'activation': random.choice(['ReLU', 'GELU']),
                           'out_channels_coefficient': random.randint(6, 10)
                       }
                       
                       genome.append(new_layer)
                   
                   elif mutation_type == 'remove' and len(genome) > 1:
                       # Remove a layer
                       genome.pop(random.randint(0, len(genome) - 1))
               
               # Add classifier back
               genome.append({
                   'layer_type': 'Classifier',
                   'activation': 'ReLU'
               })
               
               individual.genome = genome
           
           return individual

Complete Custom Block Example
==============================

Main Execution Pipeline
-----------------------

.. code-block:: python

   def main_custom_blocks_example():
       """Complete example using custom blocks in PyNAS."""
       
       # Setup
       torch.manual_seed(42)
       
       # Mock dataset for example
       class MockDataset:
           def __init__(self):
               self.num_classes = 10
       
       dataset = MockDataset()
       
       config = {
           'population_size': 16,
           'max_iterations': 8,
           'task': 'classification',
           'input_shape': (3, 224, 224)
       }
       
       # Show registered custom blocks
       print("=== Custom Block Registry ===")
       print(f"Available blocks: {custom_block_registry.list_blocks()}")
       
       for block_name in custom_block_registry.list_blocks():
           params = custom_block_registry.get_parameters(block_name)
           print(f"{block_name}: {params}")
       
       # Run evolution with custom blocks
       print(f"\n=== Starting Evolution with Custom Blocks ===")
       evolution = CustomBlockEvolution(dataset, config)
       best_individuals = evolution.evolve_with_custom_blocks()
       
       # Show results
       print(f"\n=== Evolution Results ===")
       final_best = best_individuals[-1]
       print(f"Final best fitness: {final_best.fitness:.4f}")
       print(f"Final architecture:")
       
       for i, layer in enumerate(final_best.genome):
           print(f"  Layer {i+1}: {layer}")
       
       # Test building the model
       print(f"\n=== Building Final Model ===")
       try:
           final_model = final_best.build_model()
           if final_model is not None:
               param_count = sum(p.numel() for p in final_model.parameters())
               print(f"Model built successfully!")
               print(f"Parameter count: {param_count:,}")
               print(f"Model architecture:")
               print(final_model)
           else:
               print("Failed to build model")
       except Exception as e:
           print(f"Error building model: {e}")
       
       return best_individuals
   
   if __name__ == "__main__":
       best_individuals = main_custom_blocks_example()

Expected Output
===============

The custom blocks example will produce output similar to:

.. code-block:: text

   === Custom Block Registry ===
   Available blocks: ['SeparableConvBlock', 'DilatedConvBlock', 'SelfAttentionBlock', 'ChannelAttentionBlock', 'RemoteSensingBlock', 'EdgeDetectionBlock']
   SeparableConvBlock: ['in_channels', 'out_channels', 'kernel_size', 'stride', 'padding', 'activation']
   DilatedConvBlock: ['in_channels', 'out_channels', 'dilation_rates', 'activation']
   SelfAttentionBlock: ['in_channels', 'reduction_ratio', 'activation']
   ChannelAttentionBlock: ['in_channels', 'reduction_ratio', 'use_spatial', 'activation']
   RemoteSensingBlock: ['in_channels', 'out_channels', 'spectral_reduction', 'activation']
   EdgeDetectionBlock: ['in_channels', 'out_channels', 'edge_types', 'activation']
   
   === Starting Evolution with Custom Blocks ===
   Starting evolution with 6 custom blocks
   Available blocks: ['SeparableConvBlock', 'DilatedConvBlock', 'SelfAttentionBlock', 'ChannelAttentionBlock', 'RemoteSensingBlock', 'EdgeDetectionBlock']
   
   === Generation 1 ===
   Best Individual Fitness: 0.7234
   Architecture: Sep -> Cha -> Sep -> C
   
   === Generation 8 ===
   Best Individual Fitness: 0.8456
   Architecture: Rem -> Sel -> Dil -> Cha -> C
   
   === Evolution Results ===
   Final best fitness: 0.8456
   Final architecture:
     Layer 1: {'layer_type': 'RemoteSensingBlock', 'activation': 'ReLU', 'out_channels_coefficient': 8, 'spectral_reduction': 4}
     Layer 2: {'layer_type': 'SelfAttentionBlock', 'activation': 'ReLU', 'reduction_ratio': 8}
     Layer 3: {'layer_type': 'DilatedConvBlock', 'activation': 'GELU', 'out_channels_coefficient': 10, 'dilation_rates': [1, 2, 4]}
     Layer 4: {'layer_type': 'ChannelAttentionBlock', 'activation': 'ReLU', 'reduction_ratio': 16, 'use_spatial': True}
     Layer 5: {'layer_type': 'Classifier', 'activation': 'ReLU'}
   
   === Building Final Model ===
   Model built successfully!
   Parameter count: 234,567
   Model architecture:
   Sequential(
     (0): RemoteSensingBlock(...)
     (1): SelfAttentionBlock(...)
     (2): DilatedConvBlock(...)
     (3): ChannelAttentionBlock(...)
     (4): Classifier(...)
   )

Best Practices for Custom Blocks
=================================

1. **Modular Design**: Make blocks self-contained and composable
2. **Parameter Validation**: Include proper input validation and error handling
3. **Configuration Flexibility**: Support configurable parameters for evolution
4. **Memory Efficiency**: Consider memory usage for large-scale search
5. **Documentation**: Provide clear docstrings and usage examples
6. **Testing**: Test blocks individually before integration
7. **Compatibility**: Ensure blocks work with PyTorch's standard operations

This example demonstrates how PyNAS can be extended with custom blocks, providing researchers with the flexibility to explore domain-specific architectures while leveraging the framework's evolutionary search capabilities.
