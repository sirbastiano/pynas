PyNAS: Neural Architecture Search Framework
============================================

.. image:: https://img.shields.io/badge/License-GPL%20v3-blue.svg
   :target: https://www.gnu.org/licenses/gpl-3.0
   :alt: License: GPL v3

.. image:: https://img.shields.io/badge/Python-3.9%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

PyNAS is a modular neural architecture search (NAS) framework developed by ESA Φ-lab and Little Place Lab, 
specifically designed for deployment optimization on edge devices. It leverages advanced metaheuristic strategies, 
primarily Genetic Algorithms (GA), to efficiently identify optimal deep learning architectures for constrained environments.

Key Features
------------

- **Metaheuristic Optimization**: Utilizes Genetic Algorithms (GA) for robust architecture optimization
- **Model Architecture Selection**: Automates the selection of optimal architectures for specific onboard applications  
- **Edge Device Compatibility**: Tailored for efficient deployment on various edge devices
- **Performance Metrics**: Evaluates architectures based on predefined or custom metrics relevant to edge computing
- **Customization**: Allows users to define custom constraints and requirements for model architecture
- **User-Friendly Interface**: Easy-to-use API, facilitating integration with existing projects

Use Cases
----------

- **IoT Applications**: Optimizing models for IoT devices with limited computing resources
- **Remote Sensing**: Enhancing the efficiency of models deployed in remote sensing edge devices  
- **Autonomous Vehicles**: Streamlining models for real-time processing in autonomous vehicles

Quick Start
-----------

.. code-block:: python

   import configparser
   import torch
   import pytorch_lightning as pl
   
   from pynas.core.population import Population
   from datasets.RawVessels.loader import RawVesselsDataModule
   
   # Setup data module
   root_dir = 'data/TASI/DataSAR_real_refined'
   dm = RawVesselsDataModule(root_dir, batch_size=4, num_workers=2)
   
   # Create population for genetic algorithm
   pop = Population(n_individuals=50, max_layers=5, dm=dm, max_parameters=400_000)
   
   # Initialize population  
   pop.initial_poll()
   
   # Train the generation
   pop.train_generation(task='classification', epochs=10)

Installation
------------

.. code-block:: bash

   git clone https://github.com/sirbastiano/pynas.git
   cd pynas
   pip install -e .

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:
   
   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference:
   
   api/core
   api/blocks
   api/opt
   api/train

.. toctree::
   :maxdepth: 1
   :caption: Development:
   
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

