Contributing to PyNAS
====================

We welcome contributions to PyNAS! This guide will help you get started with contributing to the project.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally::

    git clone https://github.com/your-username/PyNAS.git
    cd PyNAS

3. **Create a virtual environment**::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install development dependencies using PDM (recommended)**::

    pdm install

   Or using pip::

    pip install -e ".[dev]"

Development Setup
-----------------

The project uses several tools for development:

* **PDM** for dependency management (recommended)
* **pytest** for testing
* **black** for code formatting
* **flake8** for linting
* **mypy** for type checking

Installing Dependencies with PDM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PDM is the recommended package manager::

    pdm install

This will install all dependencies including development tools.

Installing with pip (Alternative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you don't have PDM installed::

    pip install pdm

Then install the project dependencies::

    pdm install

Code Style
----------

We follow PEP 8 and use several tools to maintain code quality:

Formatting
~~~~~~~~~~

We use **black** for code formatting::

    black pynas/
    black tests/

Linting
~~~~~~~

We use **flake8** for linting::

    flake8 pynas/
    flake8 tests/

Type Checking
~~~~~~~~~~~~~

We use **mypy** for type checking::

    mypy pynas/

Running Tests
-------------

We use **pytest** for testing. Run the test suite with::

    pytest

Run tests with coverage::

    pytest --cov=pynas

Run specific test files::

    pytest tests/test_population.py

Testing Guidelines
~~~~~~~~~~~~~~~~~~

* Write tests for all new functionality
* Maintain high test coverage (>90%)
* Use descriptive test names
* Follow the AAA pattern (Arrange, Act, Assert)
* Mock external dependencies

Example test structure::

    def test_population_initialization():
        """Test that Population initializes correctly with given parameters."""
        # Arrange
        config = PopulationConfig(
            population_size=10,
            generations=5
        )
        
        # Act
        population = Population(config)
        
        # Assert
        assert len(population.individuals) == 10
        assert population.generation == 0

Documentation
-------------

Documentation is built using Sphinx. To build the documentation locally::

    cd docs/
    make html

The documentation will be available in ``docs/build/html/``.

Documentation Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~

* Use Google-style docstrings
* Include type hints in function signatures
* Provide examples in docstrings where appropriate
* Update documentation when adding new features

Example docstring::

    def evolve_population(self, generations: int) -> Population:
        """Evolve the population for a specified number of generations.
        
        Args:
            generations: Number of generations to evolve.
            
        Returns:
            The evolved population.
            
        Raises:
            ValueError: If generations is not positive.
            
        Example:
            >>> population = Population(config)
            >>> evolved = population.evolve_population(10)
            >>> print(f"Generation: {evolved.generation}")
        """

Contributing Workflow
---------------------

1. **Create a feature branch**::

    git checkout -b feature/your-feature-name

2. **Make your changes**
   
   * Write code following the style guidelines
   * Add tests for new functionality
   * Update documentation as needed

3. **Run quality checks**::

    # Format code
    black pynas/ tests/
    
    # Check linting
    flake8 pynas/ tests/
    
    # Run type checking
    mypy pynas/
    
    # Run tests
    pytest

4. **Commit your changes**::

    git add .
    git commit -m "Add feature: brief description"

5. **Push to your fork**::

    git push origin feature/your-feature-name

6. **Create a Pull Request** on GitHub

Pull Request Guidelines
-----------------------

* **Use descriptive titles** that explain what the PR does
* **Provide a detailed description** including:
  
  * What changes were made
  * Why the changes were necessary
  * Any breaking changes
  * How to test the changes

* **Link to relevant issues** using keywords like "Fixes #123"
* **Ensure all CI checks pass**
* **Request review** from maintainers

Example PR description::

    ## Summary
    
    This PR adds support for custom fitness functions in the optimization module.
    
    ## Changes
    
    - Added `CustomFitnessFunction` class
    - Updated `Optimizer` to accept custom fitness functions
    - Added comprehensive tests
    - Updated documentation
    
    ## Testing
    
    - All existing tests pass
    - Added new tests for custom fitness functions
    - Manual testing with example custom function
    
    Fixes #45

Code Review Process
-------------------

All contributions go through code review:

1. **Automated checks** run on all PRs (tests, linting, type checking)
2. **Manual review** by maintainers
3. **Address feedback** if requested
4. **Merge** once approved

What We Look For
~~~~~~~~~~~~~~~~

* **Correctness**: Does the code work as intended?
* **Testing**: Are there adequate tests?
* **Documentation**: Is the code well-documented?
* **Style**: Does it follow our coding standards?
* **Performance**: Are there any performance implications?

Issue Reporting
---------------

When reporting issues, please include:

* **Clear description** of the problem
* **Steps to reproduce** the issue
* **Expected vs actual behavior**
* **Environment information** (OS, Python version, etc.)
* **Code snippets** or minimal examples

Use the issue templates when available.

Feature Requests
----------------

For feature requests, please:

* **Search existing issues** to avoid duplicates
* **Provide clear use cases** for the feature
* **Explain the expected behavior**
* **Consider implementation complexity**

Types of Contributions
----------------------

We welcome various types of contributions:

Code Contributions
~~~~~~~~~~~~~~~~~~

* Bug fixes
* New features
* Performance improvements
* Code refactoring

Documentation
~~~~~~~~~~~~~

* API documentation improvements
* Tutorial additions
* Example code
* README updates

Testing
~~~~~~~

* Additional test cases
* Test infrastructure improvements
* Performance benchmarks

Community
~~~~~~~~~

* Answering questions in issues
* Reviewing pull requests
* Participating in discussions

Release Process
---------------

Releases follow semantic versioning (SemVer):

* **Major version** (X.0.0): Breaking changes
* **Minor version** (X.Y.0): New features, backward compatible
* **Patch version** (X.Y.Z): Bug fixes, backward compatible

License
-------

By contributing to PyNAS, you agree that your contributions will be licensed under the project's license.

Getting Help
------------

If you need help:

* **Check the documentation** first
* **Search existing issues** on GitHub
* **Create a new issue** for questions
* **Join discussions** in the community

Thank you for contributing to PyNAS! 🚀
