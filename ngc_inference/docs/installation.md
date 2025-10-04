# NGC Inference Installation & Verification Guide

Complete guide for setting up and verifying NGC Inference for Active Inference simulations.

## Prerequisites

- **Python**: ≥ 3.10
- **Operating System**: macOS, Linux, or Windows (WSL recommended)
- **Disk Space**: ~500 MB for environment + dependencies
- **Internet**: For downloading packages

## Quick Install (Recommended)

### Step 1: Navigate to Project
```bash
cd /Users/4d/Documents/GitHub/ngc-learn/ngc_inference
```

### Step 2: Run Setup Script
```bash
./scripts/setup_environment.sh
```

This automatically:
1. Installs `uv` (if not present)
2. Creates virtual environment
3. Installs all dependencies
4. Verifies installation

### Step 3: Activate Environment
```bash
source .venv/bin/activate
```

### Step 4: Verify Installation

**Option A: Comprehensive Verification (Recommended)**
```bash
python scripts/run_ngc_inference.py
```
This runs setup, verification, all examples, and tests in one go.

**Option B: Quick Verification**
```bash
python scripts/verify_installation.py
```

Expected output: All tests pass ✓

## Manual Installation

### Option A: Using uv (Fast)

```bash
# Install uv
pip install uv

# Create virtual environment
uv venv

# Activate
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# Install package
uv pip install -e ".[dev,docs]"
```

### Option B: Using pip (Standard)

```bash
# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate    # Windows

# Upgrade pip
pip install --upgrade pip

# Install package
pip install -e ".[dev,docs]"
```

## Verification Checklist

Run through this checklist to ensure everything works:

### ✓ Package Imports
```python
import jax
import ngclearn
import ngcsimlib
from ngc_inference import __version__
print(f"NGC Inference version: {__version__}")
```

### ✓ Core Functionality
```python
from ngc_inference.core.free_energy import compute_free_energy
from ngc_inference.simulations.simple_prediction import SimplePredictionAgent

import jax.numpy as jnp
from jax import random

key = random.PRNGKey(42)
agent = SimplePredictionAgent(n_observations=10, n_hidden=20)
obs = random.normal(key, (1, 10))
beliefs, metrics = agent.infer(obs, n_steps=5)

print(f"✓ Inference successful: FE={metrics['free_energy']:.4f}")
```

### ✓ Run Tests
```bash
pytest tests/ -v
```

Expected: All tests pass

### ✓ Run Example
```bash
python scripts/run_simple_example.py
```

Expected: Training completes, results saved to logs/

## Dependencies

### Core Requirements
- `jax >= 0.4.28`: Numerical computation
- `jaxlib >= 0.4.28`: JAX backend
- `ngclearn >= 2.0.3`: Neuronal components
- `ngcsimlib >= 1.0.1`: Simulation infrastructure
- `numpy >= 1.22.0`: Array operations
- `scipy >= 1.7.0`: Scientific computing

### Utilities
- `matplotlib >= 3.8.0`: Visualization
- `pyyaml >= 6.0`: Configuration files
- `loguru >= 0.7.0`: Professional logging
- `pandas >= 2.2.3`: Data manipulation

### Development
- `pytest >= 7.4.0`: Testing framework
- `pytest-cov >= 4.1.0`: Coverage reporting
- `black >= 23.0.0`: Code formatting
- `ruff >= 0.0.270`: Linting

### Documentation
- `sphinx >= 7.0.0`: Documentation building
- `sphinx-rtd-theme >= 1.2.0`: Documentation theme
- `myst-parser >= 2.0.0`: Markdown support

## GPU Support (Optional)

To enable GPU acceleration:

### For CUDA 12:
```bash
pip install --upgrade "jax[cuda12]"
```

### For CUDA 11:
```bash
pip install --upgrade "jax[cuda11]"
```

### Verify GPU:
```python
import jax
print(f"JAX devices: {jax.devices()}")
# Should show [CudaDevice(id=0)] if GPU available
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'ngclearn'`

**Solution**:
```bash
# Ensure environment is activated
source .venv/bin/activate

# Reinstall
pip install -e ".[dev,docs]"
```

### JAX Installation Issues

**Problem**: JAX not found or incompatible

**Solution**:
```bash
# Uninstall existing
pip uninstall jax jaxlib -y

# Reinstall (CPU version)
pip install "jax[cpu]>=0.4.28"

# Or GPU version
pip install "jax[cuda12]>=0.4.28"
```

### ngclearn Version Mismatch

**Problem**: `ngclearn version too old`

**Solution**:
```bash
pip install --upgrade ngclearn ngcsimlib
```

### Permission Errors

**Problem**: `Permission denied` when installing

**Solution**:
```bash
# Don't use sudo! Use virtual environment instead
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

### pytest Not Found

**Problem**: `pytest: command not found`

**Solution**:
```bash
# Ensure dev dependencies installed
pip install -e ".[dev]"
```

## Platform-Specific Notes

### macOS
- **M1/M2 Macs**: JAX has excellent Apple Silicon support
- **Installation**: Use homebrew for Python: `brew install python@3.11`
- **Note**: May need Xcode command line tools: `xcode-select --install`

### Linux
- **Ubuntu/Debian**: `sudo apt-get install python3.11 python3.11-venv`
- **CUDA**: Ensure CUDA toolkit installed for GPU support
- **Note**: May need build essentials: `sudo apt-get install build-essential`

### Windows
- **Recommendation**: Use WSL2 (Windows Subsystem for Linux)
- **Native Windows**: Install Python from python.org
- **Note**: Some features may have issues on native Windows

## Environment Variables

Optional environment variables:

```bash
# Set JAX to use CPU only
export JAX_PLATFORM_NAME=cpu

# Set JAX to use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Set log level
export NGC_INFERENCE_LOG_LEVEL=DEBUG

# Set output directory
export NGC_INFERENCE_OUTPUT_DIR=/custom/path/logs
```

## Updating

### Update NGC Inference
```bash
cd ngc_inference
git pull  # If from git
pip install -e ".[dev,docs]" --upgrade
```

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Update Specific Package
```bash
pip install --upgrade ngclearn
```

## Uninstallation

### Remove Package
```bash
pip uninstall ngc_inference
```

### Remove Environment
```bash
deactivate  # Exit environment
rm -rf .venv  # Remove directory
```

## Next Steps

After successful installation:

1. **Read Documentation**: `docs/quickstart.md`
2. **Run Examples**: `python scripts/run_simple_example.py`
3. **Explore Tests**: `pytest tests/ -v`
4. **Try Configurations**: Edit `configs/*.yaml`
5. **Build Custom Agent**: Follow `src/ngc_inference/simulations/AGENTS.md`

## Getting Help

If you encounter issues:

1. **Check Verification**: `python scripts/verify_installation.py`
2. **Review Logs**: Check `logs/*.log` for errors
3. **Search Issues**: Look for similar problems
4. **Documentation**: Read relevant docs in `docs/`
5. **Ask Questions**: Open an issue with:
   - Python version: `python --version`
   - Package versions: `pip list | grep -E "jax|ngc"`
   - Error message (full traceback)
   - Operating system

## Installation Success Criteria

You have successfully installed NGC Inference if:

- ✓ `python scripts/verify_installation.py` passes all tests
- ✓ `pytest tests/ -v` runs without errors
- ✓ `python scripts/run_simple_example.py` completes
- ✓ All imports work without errors
- ✓ Examples generate outputs in `logs/runs/`

## Additional Resources

- **NGC Learn Docs**: https://ngc-learn.readthedocs.io/
- **JAX Documentation**: https://jax.readthedocs.io/
- **Active Inference**: Free Energy Principle literature
- **Project README**: `README.md` in root directory
- **Theory Guide**: `docs/theory.md`
- **API Reference**: `docs/API.md`

---

**Installation Time**: ~5-10 minutes
**Disk Space Required**: ~500 MB
**Supported Python**: 3.10, 3.11
**Last Updated**: 2025-10-03



