# NGC Inference: Quick Start Guide

## Installation (One Command)

```bash
cd ngc_inference
python3 scripts/run_ngc_inference.py
```

This will:
- ✅ Create virtual environment at `.venv/`
- ✅ Install all dependencies (JAX, ngclearn, etc.)
- ✅ Verify installation
- ✅ Run example simulations
- ✅ Run test suite

**Expected time**: 5-10 minutes

---

## What Gets Installed

### Core Dependencies
- **JAX** (>=0.4.28): Automatic differentiation
- **ngclearn** (>=2.0.3): Neural generative coding framework
- **ngcsimlib** (>=1.0.1): Simulation utilities
- **NumPy, SciPy, Matplotlib**: Scientific computing

### Development Tools
- **pytest**: Testing framework
- **loguru**: Advanced logging
- **black, ruff**: Code formatting
- **mypy**: Type checking

---

## After Installation

### Verify Everything Works
```bash
# Check venv was created
ls .venv/bin/python

# Check imports work
.venv/bin/python -c "import jax, ngclearn; print('Success!')"

# Run tests
.venv/bin/pytest tests/ -v
```

### Run Examples

**Simple Prediction Example**:
```bash
.venv/bin/python scripts/run_simple_example.py
```
- Demonstrates basic free energy minimization
- Creates visualizations in `logs/runs/simple_example/`

**Hierarchical Inference Example**:
```bash
.venv/bin/python scripts/run_hierarchical_example.py
```
- Demonstrates multi-level predictive coding
- Creates visualizations in `logs/runs/hierarchical_example/`

### Activate Virtual Environment (Optional)
```bash
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate  # Windows

# Now you can use python directly
python scripts/run_simple_example.py
pytest tests/ -v
```

---

## Common Issues

### Issue: scikit-image build fails
**Fix**: The setup script now handles this automatically by installing numpy first and upgrading to scikit-image>=0.21.0

### Issue: "No module named 'jax'"
**Fix**: Make sure you're using the venv Python:
```bash
.venv/bin/python  # Not system python3
```

### Issue: Tests fail to collect
**Fix**: Ensure ngc_inference is installed:
```bash
.venv/bin/pip install -e .
```

---

## Project Structure

```
ngc_inference/
├── scripts/
│   ├── run_ngc_inference.py      # Comprehensive runner
│   ├── setup_environment.sh      # Environment setup
│   ├── verify_installation.py    # Verification checks
│   ├── run_simple_example.py     # Simple example
│   └── run_hierarchical_example.py  # Hierarchical example
├── src/ngc_inference/
│   ├── core/                     # Core inference engines
│   ├── simulations/              # Example simulations
│   ├── orchestrators/            # Simulation runners
│   └── utils/                    # Utilities
├── tests/                        # Test suite
├── configs/                      # Configuration files
├── docs/                         # Documentation
└── logs/                         # Execution logs and results
```

---

## Key Files

- `pyproject.toml`: Package configuration and dependencies
- `README.md`: Detailed project documentation
- `docs/theory.md`: Theoretical background
- `docs/API.md`: API reference
- `RUNNER_IMPROVEMENTS.md`: Technical details of recent fixes
- `CHANGES_SUMMARY.md`: Executive summary of changes

---

## Development Workflow

### Make Changes
```bash
# Activate venv
source .venv/bin/activate

# Edit code
vim src/ngc_inference/core/inference.py

# Format code
black src/
ruff check src/

# Run tests
pytest tests/ -v
```

### Run Specific Tests
```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src/ngc_inference --cov-report=html
```

### Create New Simulation
```python
from ngc_inference.simulations.simple_prediction import SimplePredictionAgent
from ngc_inference.orchestrators.simulation_runner import SimulationRunner

# Create agent
agent = SimplePredictionAgent(n_observations=10, n_hidden=15, seed=42)

# Run inference
beliefs, metrics = agent.infer(observations, n_steps=20)

print(f"Free energy: {metrics['free_energy']:.4f}")
```

---

## Documentation

### Online Documentation
- **API Reference**: `docs/API.md`
- **Theory**: `docs/theory.md`
- **Quickstart**: `docs/quickstart.md`

### Generate HTML Docs
```bash
cd docs
make html
open _build/html/index.html
```

---

## Getting Help

### Check Logs
```bash
# Latest run log
ls -lt logs/run_ngc_inference_*.log | head -1

# View log
tail -f logs/run_ngc_inference_*.log
```

### Debug Mode
```python
# In any script, enable detailed logging
from ngc_inference.utils.logging_config import setup_logging
setup_logging(log_level="DEBUG", log_file="debug.log")
```

### Verify Installation
```bash
.venv/bin/python scripts/verify_installation.py
```

---

## Performance Tips

### Use JAX GPU Support (if available)
```bash
# Check JAX devices
.venv/bin/python -c "import jax; print(jax.devices())"

# JAX will automatically use GPU if available
```

### Profile Code
```python
import jax
from jax import profiler

# Start profiling
profiler.start_trace("/tmp/jax-trace")

# Your code here
agent.infer(observations, n_steps=100)

# Stop profiling
profiler.stop_trace()
```

---

## Next Steps

1. ✅ Run installation: `python3 scripts/run_ngc_inference.py`
2. ✅ Check logs for any warnings
3. ✅ Run examples to see results
4. ✅ Review documentation in `docs/`
5. ✅ Start building your own simulations!

---

## Support

- **Issues**: Check `RUNNER_IMPROVEMENTS.md` for troubleshooting
- **Questions**: Review `docs/` directory
- **Bugs**: Check logs in `logs/` directory

---

**Version**: 1.1  
**Last Updated**: October 3, 2025  
**Status**: ✅ Fully functional and tested


