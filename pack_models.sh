#!/bin/bash

conda create --name modelpacking python=3.10 -y

# Activate the environment
conda activate modelpacking


# you can also use pip to install packages that are not available via Conda
pip install torch 
pip install tensorflow 
pip install onnx

# The directory containing the model files
model_directory="./apkmodels"  # Replace with the actual path to your model files

# Function to execute the python command for a given model file
perform_operation() {
    local model_path="$1"
    local framework="$2"
    
    python pack_model.py --cover_model "$model_path" --framework "$framework" --data_format float32 --task "None"
}

# Export the function so it can be used in a subshell by find -exec
export -f perform_operation

# Find and process .tflite files
find "$model_directory" -name "*.tflite"  -exec bash -c 'perform_operation "$0" "tflite"' {} \;

# Find and process .onnx files
find "$model_directory" -name "*.onnx"  -exec bash -c 'perform_operation "$0" "onnx"' {} \;

echo "Operation completed for all model files."
