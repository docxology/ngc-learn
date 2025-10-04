#!/bin/bash
# Setup script for NGC Inference environment using uv

set -e  # Exit on error

echo "=========================================="
echo "NGC Inference Environment Setup"
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    pip install uv
fi

echo "Creating virtual environment with uv..."
uv venv

# Detect OS for activation instructions
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    ACTIVATE_CMD="source .venv/bin/activate"
    PYTHON_BIN=".venv/bin/python"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    ACTIVATE_CMD=".venv\\Scripts\\activate"
    PYTHON_BIN=".venv/Scripts/python.exe"
else
    ACTIVATE_CMD="source .venv/bin/activate"
    PYTHON_BIN=".venv/bin/python"
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# Install numpy first (required for building scikit-image)
echo "Installing build dependencies (numpy, setuptools, wheel)..."
uv pip install "numpy>=1.22.0" "setuptools>=61.0" "wheel"

# Upgrade scikit-image to avoid build issues
echo "Upgrading problematic dependencies..."
uv pip install "scikit-image>=0.21.0" --upgrade

# Install NGC Inference with all dependencies
echo "Installing NGC Inference in editable mode..."
uv pip install -e ".[dev,docs]"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Virtual environment created at: .venv"
echo ""
echo "To activate the environment, run:"
echo "  $ACTIVATE_CMD"
echo ""
echo "Or use the venv Python directly:"
echo "  $PYTHON_BIN"
echo ""
echo "To verify the installation, run:"
echo "  $PYTHON_BIN -m pytest tests/ -v"
echo ""
echo "To run a simple example, use:"
echo "  $PYTHON_BIN scripts/run_simple_example.py"
echo ""



