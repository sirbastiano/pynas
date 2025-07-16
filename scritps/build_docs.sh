#!/bin/bash
"""
Simple documentation build script for PyNAS
"""

echo "🚀 Building PyNAS Documentation"
echo "================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must be run from the PyNAS root directory"
    exit 1
fi

echo "📁 Current directory: $(pwd)"
echo "📋 Checking documentation structure..."

# Check required directories
if [ ! -d "docs/source" ]; then
    echo "❌ docs/source directory not found"
    exit 1
fi

# List documentation files
echo "📄 Documentation files found:"
find docs/source -name "*.rst" -o -name "*.py" | sort

echo ""
echo "🔧 Configuration check:"
if [ -f "docs/source/conf.py" ]; then
    echo "✅ conf.py found"
    echo "   Project: $(grep '^project = ' docs/source/conf.py | head -1)"
    echo "   Theme: $(grep '^html_theme = ' docs/source/conf.py | head -1)"
else
    echo "❌ conf.py not found"
    exit 1
fi

echo ""
echo "📚 Content structure:"
echo "   Main files: $(ls docs/source/*.rst | wc -l) RST files"
echo "   API docs: $(ls docs/source/api/*.rst | wc -l) files"
echo "   Tutorials: $(ls docs/source/tutorials/*.rst | wc -l) files"
echo "   Examples: $(ls docs/source/examples/*.rst | wc -l) files"

echo ""
echo "✅ Documentation structure validation complete!"
echo ""
echo "🏗️  To build the documentation:"
echo "   1. Install Sphinx using PDM: pdm add -d sphinx sphinx-rtd-theme"
echo "   2. Or install from requirements: pdm add -d -r docs/requirements.txt"
echo "   3. Alternative with pip: pip install -r docs/requirements.txt"
echo "   4. Build HTML: sphinx-build -b html docs/source docs/build/html"
echo "   5. Or use make: cd docs && make html"
echo ""
echo "📖 Documentation files are ready for building!"

# Create a requirements file for documentation
cat > docs/requirements.txt << EOF
sphinx>=4.0.0
sphinx-rtd-theme>=1.0.0
myst-parser>=0.18.0
sphinx-autodoc-typehints>=1.12.0
sphinx-copybutton>=0.5.0
nbsphinx>=0.8.0
EOF

echo "📦 Created docs/requirements.txt with Sphinx dependencies"
echo ""
echo "💡 PDM Usage Examples:"
echo "   Add documentation dependencies: pdm add -d -r docs/requirements.txt"
echo "   Install project with docs: pdm install --group docs"
echo ""
echo "🎉 Documentation setup complete!"
