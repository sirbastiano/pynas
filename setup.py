"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup

setup(
    name="pynas",
    version="v0.1.0",
    description="Pynattas, a powerful open-source Python package"
    + " that provides a comprehensive set of tools for Neural Architecture Search",
    long_description=open("README.md", encoding="cp437").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sirbastiano/PyNA-tta-S",
    author="Roberto Del Prete, Andrea Mazzeo, Parampuneet Thind",
    author_email="robertodelprete88@gmail.com",
    install_requires=[
        "numpy",
        "opencv-python",
        "pandas",
        "tqdm",
        "matplotlib",
        "seaborn",
        "rasterio",
        "tifffile",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
    ],
    packages=["pynas", "pynas.blocks", "pynas.train", "pynas.core", "pynas.opt"],
    python_requires=">=3.8, <4",
    project_urls={"Source": "https://github.com/sirbastiano/pynas"},
)