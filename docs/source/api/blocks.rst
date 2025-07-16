================
Blocks Module
================

The blocks module provides the fundamental neural network building blocks used in PyNAS architecture search. These modules include various convolution types, activation functions, pooling layers, and specialized heads for different tasks.

Overview
========

The blocks module is organized into several submodules:

- **Convolutions**: Various convolution blocks including classic, mobile, residual, and dense variants
- **Activations**: Activation functions like ReLU, GELU, Sigmoid, and Softmax
- **Pooling**: Max and average pooling operations
- **Heads**: Classification and segmentation heads for different tasks
- **Residual**: Residual connection utilities and implementations

.. contents:: Table of Contents
   :local:
   :depth: 2

Convolution Blocks
==================

.. automodule:: pynas.blocks.convolutions
   :members:
   :undoc-members:
   :show-inheritance:

Classic Convolution Blocks
--------------------------

.. autoclass:: pynas.blocks.convolutions.ConvAct
   :members:
   :undoc-members:
   :show-inheritance:

   Basic convolution followed by activation.

.. autoclass:: pynas.blocks.convolutions.ConvBnAct
   :members:
   :undoc-members:
   :show-inheritance:

   Convolution, batch normalization, and activation in sequence.

.. autoclass:: pynas.blocks.convolutions.ConvSE
   :members:
   :undoc-members:
   :show-inheritance:

   Convolution with Squeeze-and-Excitation attention mechanism.

Squeeze-and-Excitation Block
----------------------------

.. autoclass:: pynas.blocks.convolutions.SEBlock
   :members:
   :undoc-members:
   :show-inheritance:

   Squeeze-and-Excitation block for channel attention.

Mobile Convolution Blocks
-------------------------

.. autoclass:: pynas.blocks.convolutions.MBConv
   :members:
   :undoc-members:
   :show-inheritance:

   Mobile Inverted Bottleneck Convolution with residual connection.

.. autoclass:: pynas.blocks.convolutions.MBConvNoRes
   :members:
   :undoc-members:
   :show-inheritance:

   Mobile Inverted Bottleneck Convolution without residual connection.

CSP Blocks
----------

.. autoclass:: pynas.blocks.convolutions.CSPConvBlock
   :members:
   :undoc-members:
   :show-inheritance:

   Cross Stage Partial convolution block for efficient feature learning.

.. autoclass:: pynas.blocks.convolutions.CSPMBConvBlock
   :members:
   :undoc-members:
   :show-inheritance:

   CSP block using Mobile Inverted Bottleneck convolutions.

Dense and Residual Blocks
-------------------------

.. autoclass:: pynas.blocks.convolutions.DenseNetBlock
   :members:
   :undoc-members:
   :show-inheritance:

   DenseNet block with concatenation-based feature reuse.

.. autoclass:: pynas.blocks.convolutions.ResNetBasicBlock
   :members:
   :undoc-members:
   :show-inheritance:

   Basic ResNet block with bottleneck structure.

.. autoclass:: pynas.blocks.convolutions.ResNetBlock
   :members:
   :undoc-members:
   :show-inheritance:

   ResNet block with residual connection.

Utility Blocks
--------------

.. autoclass:: pynas.blocks.convolutions.Upsample
   :members:
   :undoc-members:
   :show-inheritance:

   Upsampling layer for increasing spatial resolution.

Activation Functions
====================

.. automodule:: pynas.blocks.activations
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pynas.blocks.activations.GELU
   :members:
   :undoc-members:
   :show-inheritance:

   Gaussian Error Linear Unit activation function.

.. autoclass:: pynas.blocks.activations.ReLU
   :members:
   :undoc-members:
   :show-inheritance:

   Rectified Linear Unit activation function.

.. autoclass:: pynas.blocks.activations.Sigmoid
   :members:
   :undoc-members:
   :show-inheritance:

   Sigmoid activation function.

.. autoclass:: pynas.blocks.activations.Softmax
   :members:
   :undoc-members:
   :show-inheritance:

   Softmax activation function for multi-class classification.

Pooling Operations
==================

.. automodule:: pynas.blocks.pooling
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pynas.blocks.pooling.AvgPool
   :members:
   :undoc-members:
   :show-inheritance:

   Average pooling operation for spatial downsampling.

.. autoclass:: pynas.blocks.pooling.MaxPool
   :members:
   :undoc-members:
   :show-inheritance:

   Max pooling operation for spatial downsampling.

Head Modules
============

.. automodule:: pynas.blocks.heads
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pynas.blocks.heads.Dropout
   :members:
   :undoc-members:
   :show-inheritance:

   Dropout layer for regularization.

.. autoclass:: pynas.blocks.heads.Classifier
   :members:
   :undoc-members:
   :show-inheritance:

   Classification head that combines encoder with adaptive pooling and linear layers.

.. autoclass:: pynas.blocks.heads.MultiInputClassifier
   :members:
   :undoc-members:
   :show-inheritance:

   Multi-input classifier for handling multiple input tensors with different shapes.

Residual Utilities
==================

.. automodule:: pynas.blocks.residual
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pynas.blocks.residual.Residual
   :members:
   :undoc-members:
   :show-inheritance:

   Generic residual connection wrapper for any PyTorch module.

.. autoclass:: pynas.blocks.residual.ResidualAdd
   :members:
   :undoc-members:
   :show-inheritance:

   Residual connection with element-wise addition.

.. autoclass:: pynas.blocks.residual.InputForward
   :members:
   :undoc-members:
   :show-inheritance:

   Module that passes input through multiple modules and aggregates results.

Utility Functions
=================

.. autofunction:: pynas.blocks.residual.add

   Element-wise addition function for residual connections.

Lambda Module
=============

.. autoclass:: pynas.blocks.Lambda
   :members:
   :undoc-members:
   :show-inheritance:

   Utility module for wrapping custom functions as PyTorch modules.

Block Configuration
===================

Block Types and Parameters
---------------------------

The following table shows the available block types and their configurable parameters:

.. list-table:: Block Types and Parameters
   :header-rows: 1
   :widths: 25 75

   * - Block Type
     - Parameters
   * - ConvAct
     - out_channels_coefficient, kernel_size, stride, padding, activation
   * - ConvBnAct  
     - out_channels_coefficient, kernel_size, stride, padding, activation
   * - ConvSE
     - out_channels_coefficient, kernel_size, stride, padding, activation
   * - MBConv
     - expansion_factor, dw_kernel_size, activation
   * - MBConvNoRes
     - expansion_factor, dw_kernel_size, activation
   * - CSPConvBlock
     - num_blocks, activation
   * - CSPMBConvBlock
     - expansion_factor, dw_kernel_size, num_blocks, activation
   * - DenseNetBlock
     - out_channels_coefficient, activation
   * - ResNetBlock
     - reduction_factor, activation
   * - AvgPool
     - kernel_size, stride
   * - MaxPool
     - kernel_size, stride
   * - Dropout
     - dropout_rate

Block Vocabulary
----------------

PyNAS uses a vocabulary system for encoding architectures:

**Convolution Blocks:**
- ``b``: ConvAct
- ``c``: ConvBnAct
- ``e``: ConvSE (with Squeeze-and-Excitation)
- ``d``: DenseNetBlock
- ``m``: MBConv
- ``n``: MBConvNoRes
- ``R``: ResNetBlock
- ``D``: Dropout

**Activation Functions:**
- ``r``: ReLU
- ``g``: GELU
- ``sg``: Sigmoid
- ``s``: Softmax

**Pooling Operations:**
- ``a``: AvgPool
- ``M``: MaxPool

Usage Examples
==============

Basic Convolution Block
-----------------------

.. code-block:: python

   from pynas.blocks.convolutions import ConvBnAct
   from pynas.blocks.activations import ReLU
   
   # Create a convolution block
   conv_block = ConvBnAct(
       in_channels=32,
       out_channels=64,
       kernel_size=3,
       stride=1,
       padding=1,
       activation=ReLU
   )
   
   # Use in forward pass
   output = conv_block(input_tensor)

Mobile Inverted Bottleneck
---------------------------

.. code-block:: python

   from pynas.blocks.convolutions import MBConv
   from pynas.blocks.activations import ReLU
   
   # Create MBConv block
   mb_block = MBConv(
       in_channels=64,
       out_channels=64,
       expansion_factor=4,
       dw_kernel_size=3,
       activation=ReLU
   )
   
   output = mb_block(input_tensor)

Classification Head
-------------------

.. code-block:: python

   from pynas.blocks.heads import Classifier
   import torch.nn as nn
   
   # Assume we have a data module with num_classes and input_shape
   class SimpleDataModule:
       def __init__(self):
           self.num_classes = 10
           self.input_shape = (3, 224, 224)
   
   # Create encoder (any PyTorch model)
   encoder = nn.Sequential(
       nn.Conv2d(3, 64, 3, padding=1),
       nn.ReLU(),
       nn.AdaptiveAvgPool2d((7, 7))
   )
   
   # Create classifier
   dm = SimpleDataModule()
   classifier = Classifier(encoder=encoder, dm=dm, verbose=True)
   
   # The classifier combines encoder + classification head
   output = classifier.model(input_tensor)

Residual Connections
--------------------

.. code-block:: python

   from pynas.blocks.residual import Residual, ResidualAdd
   from pynas.blocks.convolutions import ConvBnAct
   from pynas.blocks.activations import ReLU
   
   # Create a block with residual connection
   conv_block = ConvBnAct(64, 64, activation=ReLU)
   residual_block = ResidualAdd(conv_block)
   
   # Or use custom residual function
   custom_residual = Residual(
       block=conv_block,
       res_func=lambda x, res: x + res * 0.5  # Custom weighting
   )
   
   output = residual_block(input_tensor)

Architecture Encoding
---------------------

.. code-block:: python

   # Example architecture string: "c3r" means 3 ConvBnAct layers with ReLU
   # This gets parsed by the architecture builder into:
   
   architecture = [
       {'layer_type': 'ConvBnAct', 'activation': 'ReLU'},
       {'layer_type': 'ConvBnAct', 'activation': 'ReLU'}, 
       {'layer_type': 'ConvBnAct', 'activation': 'ReLU'}
   ]

Best Practices
==============

1. **Channel Management**: Use appropriate out_channels_coefficient values to control model complexity
2. **Activation Choice**: ReLU for general use, GELU for transformer-like architectures
3. **Pooling Strategy**: Use MaxPool for feature extraction, AvgPool before classification
4. **Residual Connections**: Essential for deep networks to avoid gradient vanishing
5. **Mobile Blocks**: Use MBConv for efficient mobile/edge deployments
6. **CSP Blocks**: Good balance between efficiency and representational power

Performance Considerations
==========================

- **MBConv**: Most parameter-efficient for mobile deployment
- **ConvBnAct**: Standard choice for balanced performance
- **CSP Blocks**: Good for avoiding gradient bottlenecks
- **DenseNet Blocks**: High memory usage but strong feature reuse
- **ResNet Blocks**: Stable training for very deep networks

The blocks in this module are designed to be modular and composable, allowing the genetic algorithm to efficiently explore different architectural combinations while maintaining compatibility across different block types.
