Changelog
=========

All notable changes to PyNAS will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~
- Comprehensive Sphinx documentation with API reference, tutorials, and examples
- Edge deployment optimization examples
- Custom block creation guides
- Advanced configuration tutorials

Changed
~~~~~~~
- Enhanced documentation structure and organization

[0.2.0] - 2024-01-15
---------------------

Added
~~~~~
- Multi-objective optimization support
- Custom fitness function capabilities
- Edge device optimization features
- Quantization and pruning utilities
- Advanced architecture search strategies
- Performance profiling tools
- Batch processing capabilities
- Model export functionality

Changed
~~~~~~~
- Improved population evolution algorithms
- Enhanced architecture builder flexibility
- Optimized memory usage during evolution
- Better error handling and logging

Fixed
~~~~~
- Memory leaks during long evolution runs
- Convergence issues with small populations
- Architecture validation edge cases

[0.1.1] - 2023-12-10
---------------------

Fixed
~~~~~
- Package installation issues
- Import path corrections
- Documentation typos
- Example code compatibility

[0.1.0] - 2023-12-01
---------------------

Added
~~~~~
- Initial release of PyNAS
- Core population-based neural architecture search
- Individual architecture representation
- Architecture builder with U-Net support
- Basic optimization algorithms (Grey Wolf, Particle Swarm)
- Training utilities and metrics
- Comprehensive block library:
  
  - Standard convolution blocks
  - Separable convolution variants
  - Attention mechanisms
  - Residual connections
  - Various activation functions
  - Pooling operations
  - Classification and segmentation heads

- Lightning integration for training
- Configuration management
- Basic examples and tutorials

Features
~~~~~~~~
- Genetic algorithm-based architecture search
- Configurable search spaces
- Multi-GPU training support
- Flexible fitness function definition
- Architecture visualization tools
- Export capabilities for deployment

Supported Architectures
~~~~~~~~~~~~~~~~~~~~~~~
- U-Net variants for segmentation
- Custom CNN architectures
- Encoder-decoder networks
- Multi-scale architectures

Optimization Algorithms
~~~~~~~~~~~~~~~~~~~~~~~
- Grey Wolf Optimizer (GWO)
- Particle Swarm Optimization (PSO)
- Custom optimizer interface

Training Features
~~~~~~~~~~~~~~~~~
- PyTorch Lightning integration
- Multiple loss functions
- Comprehensive metrics
- Early stopping
- Learning rate scheduling
- Mixed precision training

Block Library
~~~~~~~~~~~~~
- Convolution blocks (2D/3D)
- Separable convolutions
- Depthwise separable convolutions
- Dilated convolutions
- Attention blocks
- Squeeze-and-excitation
- Residual connections
- Skip connections
- Activation functions
- Normalization layers
- Pooling operations
- Dropout variants

Development Tools
~~~~~~~~~~~~~~~~~
- Comprehensive test suite
- Type hints throughout
- Documentation with Sphinx
- CI/CD pipeline
- Code quality tools

Known Issues
~~~~~~~~~~~~
- Large memory requirements for complex search spaces
- Limited support for transformer architectures
- GPU memory optimization needed for very large populations

Migration Guide
---------------

From 0.1.x to 0.2.x
~~~~~~~~~~~~~~~~~~~~

**Configuration Changes**

The configuration format has been updated to support multi-objective optimization:

.. code-block:: python

    # Old format (0.1.x)
    config = {
        'fitness_function': 'accuracy'
    }
    
    # New format (0.2.x)
    config = {
        'fitness_functions': ['accuracy', 'model_size'],
        'optimization_strategy': 'pareto'
    }

**API Changes**

Population initialization now requires explicit configuration:

.. code-block:: python

    # Old format (0.1.x)
    population = Population(size=50)
    
    # New format (0.2.x)
    config = PopulationConfig(population_size=50)
    population = Population(config)

**Import Changes**

Some modules have been reorganized:

.. code-block:: python

    # Old imports (0.1.x)
    from pynas.core import Population
    from pynas.utils import metrics
    
    # New imports (0.2.x)
    from pynas.core.population import Population
    from pynas.train.metrics import accuracy_score

Deprecation Warnings
---------------------

The following features are deprecated and will be removed in future versions:

v0.3.0 (Planned)
~~~~~~~~~~~~~~~~
- ``pynas.utils.legacy_metrics`` - Use ``pynas.train.metrics`` instead
- ``Population.evolve()`` without config - Use ``Population.evolve(config)``

v0.4.0 (Planned)
~~~~~~~~~~~~~~~~
- Direct fitness function strings - Use ``FitnessFunction`` objects
- Old configuration dictionary format - Use configuration classes

Breaking Changes
----------------

v0.2.0
~~~~~~
- Configuration format changed from dictionaries to structured classes
- Population initialization requires explicit configuration
- Some utility functions moved to different modules

v0.1.1
~~~~~~
- Import paths updated for consistency
- Some example configurations changed

Upcoming Features
-----------------

v0.3.0 (Planned)
~~~~~~~~~~~~~~~~
- Transformer architecture support
- Automated hyperparameter optimization
- Distributed evolution across multiple nodes
- Advanced visualization tools
- Model compression techniques
- ONNX export support

v0.4.0 (Planned)
~~~~~~~~~~~~~~~~
- Neural architecture morphing
- Progressive search strategies
- Reinforcement learning integration
- Cloud deployment tools
- Performance benchmarking suite

Contributing
------------

See `Contributing Guide <contributing.html>`_ for information on how to contribute to PyNAS.

Support
-------

For support and questions:

- Check the `documentation <index.html>`_
- Search `existing issues <https://github.com/esa-philab/PyNAS/issues>`_
- Create a `new issue <https://github.com/esa-philab/PyNAS/issues/new>`_

License
-------

This project is licensed under the MIT License. See the LICENSE file for details.

Authors and Acknowledgments
---------------------------

PyNAS is developed by:

- **ESA Φ-lab** - European Space Agency
- **Little Place Lab** - Research and development

Special thanks to contributors and the open-source community for their support and contributions.
