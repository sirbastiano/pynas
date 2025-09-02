#!/bin/bash
clear

PYTHON=/Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/.venv/bin/python3
SCRIPT=/Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/pyscripts/end2end.py

$PYTHON $SCRIPT --seed 42 --task classification