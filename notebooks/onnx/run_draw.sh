#!/usr/bin/env bash
set -euo pipefail
# Use the project's virtual environment Python interpreter
PYTHON_BIN="/Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/.venv/bin/python"
# Path to your TorchScript model
MODEL_PATH="/Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/notebooks/onnx/model_and_architecture_26.pt"

# Output filenames
HIER_OUT="model_hierarchy.pdf"
COMP_OUT="computation_graph.pdf"
IR_OUT="model_ir.txt"

# Example input shape (adjust channels/spatial dims if needed)
INPUT_SHAPE="1,3,256,256"

# Call the Python script
python /Data_large/marine/PythonProjects/OtherProjects/lpl-PyNas/notebooks/onnx/draw.py \
  --model "$MODEL_PATH" \
  --hier-out "$HIER_OUT" \
  --comp-out "$COMP_OUT" \
  --ir-out "$IR_OUT" \
  --input-shape "$INPUT_SHAPE" \
  --dtype float32 \
  --device cpu