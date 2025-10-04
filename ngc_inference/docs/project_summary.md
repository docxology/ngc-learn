# NGC Inference Project Summary

## Overview

**NGC Inference** is a comprehensive framework for Active Inference simulations using ngclearn, implementing variational free energy minimization for neurobiologically plausible learning and inference.

## Project Structure

```
ngc_inference/
├── pyproject.toml              # Project configuration & dependencies
├── README.md                   # Main documentation
├── PROJECT_SUMMARY.md          # This file
├── .gitignore                  # Git ignore patterns
│
├── src/ngc_inference/          # Main package source
│   ├── __init__.py
│   ├── core/                   # Core Active Inference algorithms
│   │   ├── free_energy.py      # Variational free energy computation
│   │   └── inference.py        # Base agent classes
│   ├── simulations/            # Simulation implementations
│   │   ├── simple_prediction.py      # Single-layer predictive agent
│   │   └── hierarchical_inference.py # Multi-layer hierarchical agent
│   ├── orchestrators/          # Workflow management (thin orchestration)
│   │   ├── simulation_runner.py      # Single simulation runner
│   │   └── experiment_manager.py     # Multi-run experiment manager
│   └── utils/                  # Utilities and helpers
│       ├── logging_config.py   # Professional logging with loguru
│       ├── metrics.py          # Performance metrics computation
│       └── visualization.py    # Plotting and visualization
│
├── tests/                      # Comprehensive test suite
│   ├── unit/                   # Unit tests for individual components
│   │   ├── test_free_energy.py
│   │   └── test_metrics.py
│   ├── integration/            # Integration tests
│   │   ├── test_simple_agent.py
│   │   └── test_hierarchical_agent.py
│   ├── simulations/            # End-to-end simulation tests
│   │   └── test_complete_workflow.py
│   └── test_ngclearn_integration.py  # Real ngclearn verification
│
├── configs/                    # Configuration files
│   ├── simple_prediction.yaml
│   ├── hierarchical_inference.yaml
│   └── experiment_template.yaml
│
├── scripts/                    # Utility scripts
│   ├── setup_environment.sh    # Environment setup with uv
│   ├── verify_installation.py  # Installation verification
│   ├── run_simple_example.py   # Simple prediction demo
│   └── run_hierarchical_example.py  # Hierarchical inference demo
│
├── docs/                       # Documentation
│   ├── quickstart.md          # Quick start guide
│   ├── theory.md              # Theoretical background
│   └── API.md                 # API reference
│
└── logs/                       # Runtime logs and outputs
    └── runs/                   # Simulation run outputs
```

## Key Features

### 1. **Core Active Inference**
- Variational free energy computation
- Prediction error minimization
- Expected free energy for action selection
- Gaussian entropy and KL divergence utilities

### 2. **Simulation Agents**

#### SimplePredictionAgent
- Single-layer predictive coding
- Learns generative models via free energy minimization
- Bottom-up recognition, top-down prediction
- Real ngclearn components (RateCell, GaussianErrorCell, DenseSynapse)

#### HierarchicalInferenceAgent
- Multi-layer predictive hierarchy
- Each level predicts the level below
- Hierarchical free energy minimization
- Top-down generation from abstract representations

### 3. **Orchestration**

#### SimulationRunner
- Thin orchestrator for single simulations
- Configuration management (YAML)
- Automatic result logging and visualization
- Minimal overhead, delegates to agents

#### ExperimentManager
- Parameter sweep experiments
- Grid search over hyperparameters
- Parallel execution support
- Result aggregation and analysis

### 4. **Professional Infrastructure**

#### Logging
- Structured logging with loguru
- File and console output
- Automatic rotation and retention
- Context-aware log messages

#### Testing
- pytest-based test suite
- Unit, integration, and simulation tests
- Real ngclearn integration verification
- ~100% critical path coverage

#### Visualization
- Free energy trajectory plots
- Belief state heatmaps
- Multi-metric comparison plots
- Publication-ready figures

#### Configuration
- YAML-based configuration
- Parameter grid for experiments
- Nested parameter support
- Version-controlled configs

### 5. **Documentation**
- Quick start guide with examples
- Theoretical background (Free Energy Principle)
- Complete API reference
- Inline code documentation

## Dependencies

### Core
- **Python** ≥ 3.10
- **JAX** ≥ 0.4.28 (numerical computation)
- **ngclearn** ≥ 2.0.3 (neuronal components)
- **ngcsimlib** ≥ 1.0.1 (simulation library)

### Utilities
- **numpy** ≥ 1.22.0
- **scipy** ≥ 1.7.0
- **matplotlib** ≥ 3.8.0
- **pyyaml** ≥ 6.0
- **loguru** ≥ 0.7.0

### Development
- **pytest** ≥ 7.4.0 (testing)
- **pytest-cov** (coverage)
- **black** (formatting)
- **ruff** (linting)

### Documentation
- **sphinx** ≥ 7.0.0
- **sphinx-rtd-theme**
- **myst-parser**

## Installation Methods

### Method 1: uv (Recommended)
```bash
cd ngc_inference
./scripts/setup_environment.sh
```

### Method 2: Manual uv
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,docs]"
```

### Method 3: pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Verification

```bash
python scripts/verify_installation.py
pytest tests/ -v
```

## Example Usage

### Simple Prediction
```bash
python scripts/run_simple_example.py
```

Output:
- Training curves showing free energy minimization
- Belief state visualizations
- Performance metrics (RMSE, R², etc.)

### Hierarchical Inference
```bash
python scripts/run_hierarchical_example.py
```

Output:
- Multi-level belief hierarchies
- Top-down generation demonstrations
- Layer-wise weight statistics

### Custom Simulation
```python
from ngc_inference.simulations.simple_prediction import SimplePredictionAgent
from ngc_inference.orchestrators.simulation_runner import SimulationRunner

# Configuration
config = {"agent": {...}, "training": {...}}

# Create and run
agent = SimplePredictionAgent(...)
runner = SimulationRunner(config)
results = runner.run_learning(agent, data, ...)
```

## Testing Strategy

### Test Levels
1. **Unit Tests**: Individual functions (free energy, metrics)
2. **Integration Tests**: Agent functionality with ngclearn
3. **Simulation Tests**: Complete workflows with orchestrators
4. **Verification Tests**: Real ngclearn component compatibility

### Running Tests
```bash
# All tests
pytest tests/ -v

# By category
pytest tests/unit/ -m unit
pytest tests/integration/ -m integration  
pytest tests/simulations/ -m simulation

# With coverage
pytest tests/ --cov=src/ngc_inference --cov-report=html
```

## Design Principles

### 1. **Modularity**
- Clear separation of concerns
- Composable components
- Minimal dependencies between modules

### 2. **Neurobiological Plausibility**
- Uses real ngclearn components
- Predictive coding architecture
- Local learning rules

### 3. **Thin Orchestration**
- Orchestrators coordinate, don't implement
- Domain logic in agents
- Configuration-driven workflows

### 4. **Professional Standards**
- Type hints throughout
- Comprehensive documentation
- Test-driven development
- Structured logging

### 5. **Reproducibility**
- Seed control for randomness
- Version-pinned dependencies
- Configuration versioning
- Complete experiment tracking

## Theoretical Foundation

### Free Energy Principle
- Biological systems minimize variational free energy
- Free Energy = Accuracy + Complexity
- Unifies perception, action, and learning

### Predictive Coding
- Hierarchical generative models
- Prediction error minimization
- Top-down predictions, bottom-up errors

### Active Inference
- Action selection via expected free energy
- Balances pragmatic and epistemic value
- Unified perception-action framework

## Future Extensions

### Potential Additions
1. **Action selection** (active inference with policies)
2. **Temporal dynamics** (sequential inference)
3. **Multi-modal learning** (vision + proprioception)
4. **Neuromorphic deployment** (Lava integration)
5. **Advanced visualizations** (interactive dashboards)

### Research Directions
1. Deep active inference
2. Hierarchical reinforcement learning
3. Continual learning with free energy
4. Neuromorphic active inference

## Citation

If you use this code, please cite ngclearn:

```bibtex
@article{Ororbia2022,
  title={The neural coding framework for learning generative models},
  author={Ororbia, Alexander and Kifer, Daniel},
  journal={Nature Communications},
  year={2022},
  volume={13},
  pages={2064}
}
```

## License

BSD-3-Clause (following ngclearn)

## Contact

For questions or issues:
- Check documentation in `docs/`
- Review ngclearn docs: https://ngc-learn.readthedocs.io/
- NAC Lab: https://www.cs.rit.edu/~ago/nac_lab.html

---

**Status**: Production-ready, comprehensive Active Inference framework
**Version**: 0.1.0
**Last Updated**: 2025-10-03



