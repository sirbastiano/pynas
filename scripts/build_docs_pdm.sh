#!/bin/bash
"""
PDM-based documentation build script for PyNAS
"""

echo "🚀 Building PyNAS Documentation with PDM"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Must be run from the PyNAS root directory"
    exit 1
fi

# Check if PDM is installed
if ! command -v pdm &> /dev/null; then
    echo "⚠️  PDM not found. Installing PDM..."
    pip install pdm
fi

echo "📁 Current directory: $(pwd)"
echo "🔧 Using PDM for dependency management"

# Install documentation dependencies using PDM
echo "📦 Installing documentation dependencies with PDM..."
pdm add -d sphinx>=4.0.0 sphinx-rtd-theme>=1.0.0 myst-parser>=0.18.0 sphinx-autodoc-typehints>=1.12.0 sphinx-copybutton>=0.5.0 nbsphinx>=0.8.0

# Ensure project dependencies are installed
echo "📦 Installing project dependencies..."
pdm install

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
echo "🏗️  Building documentation with Sphinx..."
cd docs/

# Build HTML documentation
if command -v make &> /dev/null; then
    echo "📖 Building with make..."
    make html
    BUILD_SUCCESS=$?
else
    echo "📖 Building with sphinx-build..."
    pdm run sphinx-build -b html source build/html
    BUILD_SUCCESS=$?
fi

cd ..

if [ $BUILD_SUCCESS -eq 0 ]; then
    echo ""
    echo "✅ Documentation build completed successfully!"
    echo ""
    echo "📖 View documentation:"
    echo "   Local file: file://$(pwd)/docs/build/html/index.html"
    echo "   Command: open docs/build/html/index.html"
    echo ""
    echo "🔍 Quick validation:"
    if [ -f "docs/build/html/index.html" ]; then
        echo "   ✅ Main page generated"
    else
        echo "   ❌ Main page missing"
    fi
    
    if [ -d "docs/build/html/api" ]; then
        echo "   ✅ API documentation generated"
    else
        echo "   ❌ API documentation missing"
    fi
    
    if [ -d "docs/build/html/tutorials" ]; then
        echo "   ✅ Tutorials generated"
    else
        echo "   ❌ Tutorials missing"
    fi
    
    if [ -d "docs/build/html/examples" ]; then
        echo "   ✅ Examples generated"
    else
        echo "   ❌ Examples missing"
    fi
else
    echo ""
    echo "❌ Documentation build failed!"
    echo "Check the build output above for errors."
    exit 1
fi

echo ""
echo "🎉 PyNAS Documentation build complete with PDM!"
echo ""
echo "📋 Next steps:"
echo "   1. Review: open docs/build/html/index.html"
echo "   2. Deploy: Configure hosting (GitHub Pages, Read the Docs, etc.)"
echo "   3. Share: Announce documentation availability"
