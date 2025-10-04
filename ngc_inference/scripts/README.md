# NGC Inference Scripts

Utility scripts for environment setup, verification, and example simulations.

## Scripts

### `run_ngc_inference.py` ⭐
**Purpose**: Comprehensive runner that executes all setup, verification, and examples

**Usage**:
```bash
python scripts/run_ngc_inference.py
```

**Actions**:
1. Sets up environment (calls setup_environment.sh)
2. Verifies installation (calls verify_installation.py)
3. Runs simple prediction example
4. Runs hierarchical inference example
5. Runs full test suite
6. Generates comprehensive summary report
7. Logs everything to timestamped log file

**Output**:
- Console output with real-time progress
- Detailed log file in `logs/run_ngc_inference_TIMESTAMP.log`
- JSON results file in `logs/run_results_TIMESTAMP.json`
- All example outputs in `logs/runs/`

**Expected Runtime**: ~5-10 minutes (depending on system)

**Success Criteria**:
- ✓ Environment setup completes
- ✓ All imports successful
- ✓ Both examples run without errors
- ✓ Test suite passes
- ✓ Exit code 0

**Use This For**:
- First-time setup verification
- Confirming everything works after updates
- Demonstrating full functionality
- Automated CI/CD pipelines

---

### `setup_environment.sh`
**Purpose**: Automated environment setup using uv

**Usage**:
```bash
cd ngc_inference
./scripts/setup_environment.sh
```

**Actions**:
1. Checks if uv is installed (installs if missing)
2. Creates virtual environment with uv
3. Activates virtual environment
4. Installs ngc_inference in editable mode with dev/docs dependencies
5. Displays next steps

**Requirements**:
- Python ≥ 3.10
- pip (for uv installation)
- Internet connection

**Output**:
- `.venv/` directory with isolated environment
- All dependencies installed
- Ready-to-use development environment

---

### `verify_installation.py`
**Purpose**: Comprehensive installation verification

**Usage**:
```bash
python scripts/verify_installation.py
```

**Tests**:
1. **Package Imports**: All required packages (jax, ngclearn, numpy, etc.)
2. **NGC Inference Package**: All modules import correctly
3. **ngclearn Components**: Real component creation and wiring
4. **Basic Functionality**: Free energy computation, agent inference

**Output**:
- ✓/✗ status for each test
- Detailed error messages if failures
- Summary with pass/fail counts
- Exit code 0 (success) or 1 (failure)

**Example Output**:
```
==================================================
NGC Inference Installation Verification
==================================================

Testing imports...
  ✓ jax
  ✓ numpy
  ✓ ngclearn
  ✓ loguru
✓ All imports successful

Testing NGC Inference...
  ✓ ngc_inference version: 0.1.0
  ✓ Free energy module
  ✓ Simple prediction agent
✓ NGC Inference package verified

Testing basic functionality...
  ✓ Free energy computation: 12.3456
  ✓ Agent inference: FE=8.7654
✓ Basic functionality verified

==================================================
Verification Summary
==================================================
✓ PASS: Package Imports
✓ PASS: NGC Inference Package
✓ PASS: ngclearn Components
✓ PASS: Basic Functionality

🎉 All verification tests passed!
Installation is successful and ready to use.
```

---

### `run_simple_example.py`
**Purpose**: Demonstrates SimplePredictionAgent with complete workflow

**Usage**:
```bash
python scripts/run_simple_example.py
```

**Features**:
- Generates sinusoidal training data
- Creates SimplePredictionAgent
- Uses SimulationRunner for orchestration
- Trains agent (50 epochs)
- Tests inference on new data
- Saves all results and visualizations

**Outputs** (in `logs/runs/simple_example/`):
- `config.yaml`: Configuration used
- `learning_metrics.json`: Training metrics
- `inference_metrics.json`: Inference metrics
- `learning_curve.png`: Training loss over epochs
- `free_energy.png`: Free energy trajectory
- `beliefs.png`: Belief states visualization

**Key Code Sections**:
```python
# Data generation
data = generate_sinusoidal_data(key, n_samples=50, n_features=10)

# Agent creation
agent = SimplePredictionAgent(
    n_observations=10,
    n_hidden=15,
    learning_rate=0.01
)

# Training with orchestrator
runner = SimulationRunner(config, output_dir="logs/runs/simple_example")
results = runner.run_learning(agent, data, n_epochs=50)

# Testing inference
beliefs, metrics = agent.infer(test_obs, n_steps=30)
```

**Expected Runtime**: ~1-2 minutes

---

### `run_hierarchical_example.py`
**Purpose**: Demonstrates HierarchicalInferenceAgent with multi-level inference

**Usage**:
```bash
python scripts/run_hierarchical_example.py
```

**Features**:
- Generates hierarchically structured data (multi-scale)
- Creates 3-layer HierarchicalInferenceAgent
- Trains agent (80 epochs)
- Tests inference on new data
- Demonstrates top-down generation
- Saves results and weight statistics

**Outputs** (in `logs/runs/hierarchical_example/`):
- `config.yaml`: Configuration
- `learning_metrics.json`: Training metrics with all layer weights
- `learning_curve.png`: Hierarchical loss over epochs

**Architecture**:
```
Layer 3 (8 units) - Abstract
    ↓
Layer 2 (12 units) - Mid-level
    ↓
Layer 1 (15 units) - Low-level
    ↓
Observations (10 units) - Concrete
```

**Key Code Sections**:
```python
# Hierarchical data
data = generate_hierarchical_data(key, n_samples=80, n_features=10)

# Agent creation
agent = HierarchicalInferenceAgent(
    layer_sizes=[10, 15, 12, 8],  # Bottom to top
    learning_rate=0.005
)

# Training
results = runner.run_learning(agent, data, n_epochs=80)

# Top-down generation
top_state = random.normal(key, (1, 8))
generated = agent.generate(top_state)
```

**Expected Runtime**: ~3-5 minutes

---

## Usage Patterns

### Quick Start

**Option 1: Comprehensive Runner (Recommended)**
```bash
# Run everything at once
python scripts/run_ngc_inference.py
```

**Option 2: Step by Step**
```bash
# Setup environment
./scripts/setup_environment.sh

# Verify installation
python scripts/verify_installation.py

# Run simple example
python scripts/run_simple_example.py

# Run hierarchical example
python scripts/run_hierarchical_example.py
```

### Development Workflow
```bash
# Activate environment
source .venv/bin/activate

# Run verification before coding
python scripts/verify_installation.py

# Make changes to code
# ...

# Test changes
pytest tests/ -v

# Run examples to verify
python scripts/run_simple_example.py
```

### Debugging Examples
```bash
# Run with verbose output
python scripts/run_simple_example.py 2>&1 | tee debug.log

# Check outputs
ls -la logs/runs/simple_example/

# Inspect results
python -c "
import json
with open('logs/runs/simple_example/learning_metrics.json') as f:
    metrics = json.load(f)
    print(f'Final loss: {metrics["final_loss"]}')"
```

## Creating Custom Scripts

Template for new example script:

```python
#!/usr/bin/env python3
"""
My custom Active Inference example.

Description of what this demonstrates.
"""

import jax.numpy as jnp
from jax import random

from ngc_inference.simulations.simple_prediction import SimplePredictionAgent
from ngc_inference.orchestrators.simulation_runner import SimulationRunner
from ngc_inference.utils.logging_config import setup_logging


def main():
    """Run custom example."""
    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/my_example.log")
    
    print("=" * 70)
    print("My Custom Example")
    print("=" * 70)
    
    # Configuration
    config = {
        "experiment": "my_experiment",
        "agent": {...},
        "training": {...}
    }
    
    # Initialize
    key = random.PRNGKey(42)
    
    # Create agent and data
    agent = SimplePredictionAgent(**config["agent"])
    data = ...  # Generate or load data
    
    # Run simulation
    runner = SimulationRunner(config, output_dir="logs/runs/my_example")
    results = runner.run_learning(agent, data, ...)
    
    # Report results
    print(f"Final loss: {results['final_loss']:.4f}")
    print("✓ Example complete!")


if __name__ == "__main__":
    main()
```

## Script Development Guidelines

1. **Shebang**: Include `#!/usr/bin/env python3`
2. **Docstring**: Clear description at top
3. **Imports**: Organized and complete
4. **Main Function**: Wrap logic in `main()`
5. **Logging**: Setup with appropriate level
6. **Configuration**: Use dictionaries or YAML
7. **Output**: Clear progress messages
8. **Error Handling**: Graceful failure with messages
9. **Documentation**: Inline comments for complex sections

## Executable Permissions

Make scripts executable:
```bash
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

## Environment Variables

Useful environment variables:
```bash
export NGC_INFERENCE_LOG_LEVEL=DEBUG
export NGC_INFERENCE_OUTPUT_DIR=/custom/path
export CUDA_VISIBLE_DEVICES=0  # GPU selection
```

## Performance Tips

**For Faster Execution**:
- Reduce `n_epochs` in examples
- Reduce `n_inference_steps`
- Use smaller `n_samples`
- Enable JAX GPU if available

**For GPU Usage**:
```python
import jax
print(f"JAX devices: {jax.devices()}")
# Should show GPU if available
```

## Troubleshooting

**Script Won't Run**:
- Check Python version: `python --version` (need ≥ 3.10)
- Activate environment: `source .venv/bin/activate`
- Verify installation: `python scripts/verify_installation.py`

**Import Errors**:
- Ensure you're in ngc_inference directory
- Check PYTHONPATH: `echo $PYTHONPATH`
- Reinstall: `pip install -e .`

**Slow Execution**:
- First run compiles JAX functions (slow)
- Subsequent runs much faster (JIT cached)
- Use smaller examples for testing

**Output Not Saved**:
- Check `logs/` directory exists
- Verify write permissions
- Check disk space

## Integration with IDEs

### VS Code
Add to launch.json:
```json
{
    "name": "Run Simple Example",
    "type": "python",
    "request": "launch",
    "program": "${workspaceFolder}/scripts/run_simple_example.py",
    "console": "integratedTerminal"
}
```

### PyCharm
Right-click script → Run 'run_simple_example'

## Maintenance

Scripts are maintained alongside package code:
- Update when API changes
- Keep examples simple and clear
- Test scripts before releases
- Document any dependencies



