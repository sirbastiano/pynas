#!/bin/bash

conda activate pynas && python3 /Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/nas_seg_burned.py --mutation_probability 0.01
conda activate pynas && python3 /Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/nas_seg_burned.py --mutation_probability 0.1
conda activate pynas && python3 /Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/nas_seg_burned.py --mutation_probability 0.2
conda activate pynas && python3 /Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/nas_seg_burned.py --mutation_probability 0.3