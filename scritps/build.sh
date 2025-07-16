#!/bin/bash

# Exit on error
set -e
cd /Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas

# Define variables
PACKAGE_NAME="pynas"
DIST_DIR="dist"
LOG_FILE="build.log"

# Initialize the log file
echo "Starting build process for $PACKAGE_NAME at $(date)" > "$LOG_FILE"

# Function to log and execute commands
log_and_run() {
    echo "Running: $*" | tee -a "$LOG_FILE"
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

# Step 1: Clean previous builds
log_and_run echo "Cleaning previous builds..."
log_and_run rm -rf "$DIST_DIR" build *.egg-info

# Step 2: Build the package
log_and_run echo "Building the package..."
log_and_run python3 setup.py sdist bdist_wheel

# Step 3: Check the package
log_and_run echo "Checking the package with twine..."
log_and_run twine check "$DIST_DIR"/*

# Step 4: Upload the package to PyPI
log_and_run echo "Uploading the package to PyPI..."
log_and_run twine upload --repository-url https://test.pypi.org/legacy/ "$DIST_DIR"/* --verbose
# Uncomment the following line to upload to the official PyPI
# log_and_run twine upload "$DIST_DIR"/*

# Step 5: Success message
log_and_run echo "Package $PACKAGE_NAME has been successfully built and uploaded!"