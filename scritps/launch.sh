#!/bin/bash

# Remove the folders models_traced and logs if they exist
rm -rf  logs lightning_logs

conda activate pynas && python3 /Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/nas_seg_burned.py