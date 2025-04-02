#!/bin/bash

# Remove the folders models_traced and logs if they exist
rm -rf models_traced logs


conda activate pynas && python3 nas_seg_burned.py
