Installation
============

System Requirements
-------------------

PyNAS requires the following system dependencies:

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)
- At least 8GB RAM
- Linux, macOS, or Windows

Python Dependencies
-------------------

PyNAS depends on several key Python packages:

- PyTorch 2.0.1
- PyTorch Lightning
- NumPy < 2.0.0
- Pandas
- OpenCV Python
- Matplotlib
- Seaborn
- Scikit-learn
- Scikit-image
- TQDM

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/sirbastiano/pynas.git
   cd pynas

2. Create a virtual environment:

.. code-block:: bash

   # Using conda
   conda create -n pynas python=3.9
   conda activate pynas
   
   # OR using venv
   python -m venv pynas_env
   source pynas_env/bin/activate  # On Windows: pynas_env\\Scripts\\activate

3. Install the package in development mode:

.. code-block:: bash

   pip install -e .

Using PDM (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

PDM is the recommended package manager for PyNAS development:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/esa-philab/PyNAS.git
   cd PyNAS
   
   # Install using PDM
   pdm install

Using pip
~~~~~~~~~

Alternatively, you can use pip:

.. code-block:: bash

   pip install pynas

Verify Installation
-------------------

To verify that PyNAS is installed correctly, run the following Python code:

.. code-block:: python

   import pynas
   from pynas.core.individual import Individual
   from pynas.core.population import Population
   
   print(f"PyNAS version: {pynas.__version__ if hasattr(pynas, '__version__') else '0.1.0'}")
   print("PyNAS installed successfully!")

GPU Setup
----------

For optimal performance, ensure CUDA is properly installed:

1. Install CUDA Toolkit (version 11.0 or higher recommended)
2. Verify PyTorch can access GPU:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'pynas'**

Ensure you've activated the correct virtual environment and installed the package.

**CUDA out of memory**

Reduce batch size or use a smaller population size in your genetic algorithm configuration.

**Missing dependencies**

If you encounter missing dependencies, install them manually:

.. code-block:: bash

   pip install torch==2.0.1 pytorch-lightning numpy pandas opencv-python

Development Installation
------------------------

For development purposes, install additional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"
   
This includes testing and documentation dependencies.
