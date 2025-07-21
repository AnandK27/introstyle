#!/bin/bash

# IntroStyle Installation Script
# This script helps install the required dependencies for IntroStyle

set -e  # Exit on any error

echo "=================================================="
echo "IntroStyle Installation Script"
echo "=================================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
echo "Python version: $python_version"

if [[ $(echo "$python_version 3.8" | awk '{print ($1 >= $2)}') == 0 ]]; then
    echo "Error: Python 3.8 or higher is required"
    exit 1
fi

# Check if pip is installed
echo "Checking pip installation..."
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed"
    exit 1
fi

# Install PyTorch (with CUDA support if available)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other dependencies..."
pip install diffusers transformers accelerate
pip install numpy Pillow tqdm scipy matplotlib

# Install optional dependencies
echo "Installing optional dependencies..."
pip install xformers || echo "Warning: xformers installation failed (optional)"

# Verify installation
echo "Verifying installation..."
python3 -c "
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image
print('✓ All dependencies installed successfully!')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name()}')
"

echo "=================================================="
echo "Installation completed successfully!"
echo ""
echo "You can now run IntroStyle feature extraction:"
echo "  python extract_features.py --help"
echo ""
echo "Or try the example script:"
echo "  python example.py"
echo "=================================================="
