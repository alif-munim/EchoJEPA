#!/bin/bash
# Setup CPU-only PyTorch virtual environment for classification

set -euo pipefail

echo "========================================="
echo "Setting up CPU-only PyTorch environment"
echo "========================================="

# Load required modules
module purge
module load gcc/12.1.0
module load python/3.12.3

# Define paths
VENV_DIR="/gpfs/data/whitney-lab/echo-FM/CODE/adapted_cri/torch_cpu_venv"

# Create virtual environment if it doesn't exist
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at: $VENV_DIR"
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists at: $VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

echo ""
echo "Installing CPU-only PyTorch and dependencies..."
echo "This may take several minutes..."
echo ""

# Upgrade pip
pip install --upgrade pip

# Install CPU-only PyTorch from PyTorch repository
# Using index-url to get CPU-only builds
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other required packages
pip install Pillow

echo ""
echo "========================================="
echo "Installation complete!"
echo "========================================="
echo ""
echo "Virtual environment: $VENV_DIR"
echo ""
echo "To use this environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""

# Verify installation
echo "Verifying PyTorch installation..."
python3 << 'EOF'
import torch
print(f"✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
print(f"✓ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
x = torch.randn(5, 5)
print(f"✓ Test tensor created on: {x.device}")
print("\n✓ CPU-only PyTorch is working correctly!")
EOF

echo ""
echo "========================================="
echo "Setup successful!"
echo "========================================="
