# PyNAS Documentation

This directory contains the complete Sphinx documentation for PyNAS (Neural Architecture Search framework).

## 📁 Structure

```
docs/
├── source/                    # Documentation source files
│   ├── conf.py               # Sphinx configuration
│   ├── index.rst             # Main documentation page
│   ├── installation.rst      # Installation guide
│   ├── quickstart.rst        # Quick start guide
│   ├── contributing.rst      # Contribution guidelines
│   ├── changelog.rst         # Version history
│   ├── api/                  # API reference documentation
│   │   ├── index.rst         # API overview
│   │   ├── core.rst          # Core modules (Population, Individual, etc.)
│   │   ├── blocks.rst        # Neural network blocks
│   │   ├── opt.rst           # Optimization algorithms
│   │   └── train.rst         # Training utilities
│   ├── tutorials/            # Step-by-step tutorials
│   │   ├── index.rst         # Tutorials overview
│   │   ├── basic_nas.rst     # Basic NAS concepts
│   │   ├── advanced_config.rst   # Advanced configuration
│   │   ├── custom_architectures.rst  # Custom architecture creation
│   │   └── edge_deployment.rst     # Edge device deployment
│   └── examples/             # Complete working examples
│       ├── index.rst         # Examples overview
│       ├── vessel_detection.rst    # SAR vessel detection example
│       ├── remote_sensing.rst      # Remote sensing applications
│       ├── edge_optimization.rst   # Edge optimization example
│       └── custom_blocks.rst       # Custom block creation
├── build/                    # Generated documentation (after build)
├── requirements.txt          # Documentation dependencies
├── Makefile                  # Build automation
└── make.bat                  # Windows build script
```

## 🚀 Building the Documentation

### Prerequisites

1. **Install Sphinx and dependencies using PDM:**
   ```bash
   pdm add -d sphinx>=4.0.0 sphinx-rtd-theme>=1.0.0 myst-parser>=0.18.0 sphinx-autodoc-typehints>=1.12.0 sphinx-copybutton>=0.5.0 nbsphinx>=0.8.0
   ```

   Or install from requirements file:
   ```bash
   pdm add -d -r docs/requirements.txt
   ```

   Alternative with pip (if PDM is not available):
   ```bash
   pip install -r docs/requirements.txt
   ```

2. **Ensure PyNAS is installed:**
   ```bash
   pdm install
   ```

### Build Commands

**Option 1: Using Make (recommended)**
```bash
cd docs/
make html
```

**Option 2: Using sphinx-build directly**
```bash
sphinx-build -b html docs/source docs/build/html
```

**Option 3: Using the build script**
```bash
./build_docs.sh
```

### Viewing the Documentation

After building, open `docs/build/html/index.html` in your web browser.

## 📚 Documentation Sections

### 🏁 Getting Started
- **Installation Guide**: System requirements, dependencies, GPU setup
- **Quick Start**: Basic usage examples and workflow
- **API Reference**: Complete module and function documentation

### 📖 Tutorials
- **Basic NAS**: Introduction to neural architecture search concepts
- **Advanced Configuration**: Custom fitness functions, multi-objective optimization
- **Custom Architectures**: Creating and integrating custom building blocks
- **Edge Deployment**: Optimization for resource-constrained devices

### 💡 Examples
- **Vessel Detection**: Complete SAR vessel detection pipeline
- **Remote Sensing**: Satellite imagery analysis applications
- **Edge Optimization**: Mobile device deployment optimization
- **Custom Blocks**: Creating custom neural network components

### 🛠️ Development
- **Contributing Guide**: How to contribute to PyNAS
- **Changelog**: Version history and breaking changes

## 🔧 Configuration

The documentation is configured in `docs/source/conf.py` with:

- **Theme**: Read the Docs theme
- **Extensions**: 
  - `sphinx.ext.autodoc` - Automatic API documentation
  - `sphinx.ext.napoleon` - Google/NumPy docstring support
  - `sphinx.ext.viewcode` - Source code links
  - `sphinx.ext.intersphinx` - Cross-project references
- **Auto-documentation**: Automatic generation from docstrings
- **Cross-references**: Links to PyTorch and Lightning documentation

## 📝 Writing Documentation

### RST Syntax
The documentation uses reStructuredText (RST) format. Key syntax:

```rst
Section Title
=============

Subsection
----------

**Bold text**
*Italic text*
``Code inline``

.. code-block:: python

   # Code block
   import pynas

.. note::
   This is a note box.

.. warning::
   This is a warning box.
```

### Adding New Content

1. **New Tutorial**: Add `.rst` file to `tutorials/` and update `tutorials/index.rst`
2. **New Example**: Add `.rst` file to `examples/` and update `examples/index.rst`
3. **API Changes**: Update relevant files in `api/` directory

### Code Examples

Include executable code examples:

```python
import torch
from pynas.core import Population

# Create population
pop = Population(n_individuals=10)
pop.initial_poll()
```

## 🧪 Testing Documentation

### Validate Structure
```bash
python validate_docs.py
```

### Check for Broken Links
```bash
sphinx-build -b linkcheck docs/source docs/build/linkcheck
```

### Test Build
```bash
make html 2>&1 | tee build.log
```

## 🚀 Deployment

### Local Development
- Build and serve locally for development
- Use `make livehtml` for auto-rebuild on changes

### CI/CD Integration
The documentation can be automatically built and deployed using:
- GitHub Actions
- Read the Docs
- GitLab CI

### Read the Docs Setup
1. Connect repository to Read the Docs
2. Configure build settings:
   - Python version: 3.9+
   - Requirements file: `docs/requirements.txt`
   - Configuration file: `docs/source/conf.py`

## 🎯 Best Practices

### Content Guidelines
- **Clear Structure**: Use consistent heading hierarchy
- **Code Examples**: Include working, tested examples
- **Cross-References**: Link between related sections
- **Up-to-Date**: Keep examples current with latest API

### Technical Guidelines
- **Docstrings**: Use Google-style docstrings with type hints
- **Images**: Store in `docs/source/_static/images/`
- **Downloads**: Store files in `docs/source/_static/downloads/`
- **External Links**: Use intersphinx for external documentation

### Maintenance
- **Regular Updates**: Keep documentation current with code changes
- **Link Checking**: Regularly test for broken links
- **User Feedback**: Incorporate user suggestions and corrections

## 🆘 Troubleshooting

### Common Issues

**Build Errors:**
- Check Python path in `conf.py`
- Ensure all dependencies are installed
- Verify RST syntax with linter

**Missing API Documentation:**
- Check module imports in Python path
- Verify autodoc configuration
- Ensure docstrings are properly formatted

**Theme Issues:**
- Reinstall theme: `pip install --upgrade sphinx-rtd-theme`
- Clear build cache: `rm -rf docs/build/`

### Getting Help
- Check Sphinx documentation: https://www.sphinx-doc.org/
- RST syntax guide: https://docutils.sourceforge.io/rst.html
- Community forums and Stack Overflow

## 📞 Contact

For documentation-specific questions:
- Create an issue on GitHub
- Contact the development team
- Check existing documentation discussions

---

**Happy Documenting! 📚✨**
