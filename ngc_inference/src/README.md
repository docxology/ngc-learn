# NGC Inference Source Code

This directory contains the main implementation of the NGC Inference framework for Active Inference simulations using ngclearn.

## Structure

```
src/ngc_inference/
├── __init__.py              # Package initialization and exports
├── core/                    # Core Active Inference algorithms
│   ├── __init__.py
│   ├── free_energy.py      # Variational free energy computation
│   └── inference.py        # Base agent classes
├── simulations/            # Simulation implementations
│   ├── __init__.py
│   ├── simple_prediction.py      # Single-layer predictive agent
│   └── hierarchical_inference.py # Multi-layer hierarchical agent
├── orchestrators/          # Workflow management
│   ├── __init__.py
│   ├── simulation_runner.py     # Single simulation orchestrator
│   └── experiment_manager.py    # Multi-run experiment manager
└── utils/                  # Utilities and helpers
    ├── __init__.py
    ├── logging_config.py   # Professional logging with loguru
    ├── metrics.py          # Performance metrics computation
    └── visualization.py    # Plotting and visualization
```

## Core Modules

### `core/free_energy.py`
Implements variational free energy calculations following the Free Energy Principle:
- `compute_free_energy()`: VFE = Accuracy + Complexity
- `compute_prediction_error()`: Precision-weighted errors
- `compute_expected_free_energy()`: For action selection
- `compute_gaussian_entropy()`: Entropy of beliefs
- `compute_kl_divergence()`: KL between distributions

All functions are JIT-compiled for performance.

### `core/inference.py`
Base classes for Active Inference agents:
- `VariationalInferenceAgent`: Perception through free energy minimization
- `ActiveInferenceAgent`: Adds action selection via expected free energy

Both use real ngclearn components (RateCell, GaussianErrorCell, DenseSynapse).

## Simulation Agents

### `simulations/simple_prediction.py`
**SimplePredictionAgent**: Single-layer predictive coding
- Architecture: Input → Error → Hidden → Prediction
- Learning: Minimize free energy through gradient descent
- Use case: Basic sensory prediction, feature learning

### `simulations/hierarchical_inference.py`
**HierarchicalInferenceAgent**: Multi-layer predictive hierarchy
- Architecture: Stack of prediction-error units
- Learning: Hierarchical free energy minimization
- Use case: Deep inference, abstract representations

## Orchestration

### `orchestrators/simulation_runner.py`
**SimulationRunner**: Thin orchestrator for single simulations
- Configuration management (YAML)
- Automatic result logging (JSON + NPY)
- Visualization generation (PNG)
- Progress tracking and reporting

Design: Delegates to agents, doesn't implement domain logic.

### `orchestrators/experiment_manager.py`
**ExperimentManager**: Parameter sweep experiments
- Grid search over hyperparameters
- Parallel execution support
- Result aggregation and analysis
- Best run identification

## Utilities

### `utils/logging_config.py`
Professional logging with loguru:
- Structured output (file + console)
- Automatic rotation and retention
- Context-aware messages
- Configurable log levels

### `utils/metrics.py`
Performance metrics computation:
- `InferenceMetrics`: Container for tracking over time
- `compute_metrics()`: Comprehensive inference metrics
- `compute_rmse()`, `compute_mae()`, `compute_r2_score()`: Standard metrics

### `utils/visualization.py`
Visualization utilities:
- `plot_free_energy()`: Free energy trajectories
- `plot_beliefs()`: Belief state heatmaps
- `plot_metrics_comparison()`: Multi-metric plots

All plots are publication-ready (300 DPI, proper labels).

## Usage Patterns

### Basic Inference
```python
from ngc_inference.simulations.simple_prediction import SimplePredictionAgent

agent = SimplePredictionAgent(n_observations=10, n_hidden=20)
beliefs, metrics = agent.infer(observations, n_steps=30)
```

### Learning
```python
training_results = agent.learn(
    observations,
    n_epochs=100,
    n_inference_steps=20,
    verbose=True
)
```

### Orchestrated Simulation
```python
from ngc_inference.orchestrators.simulation_runner import SimulationRunner

runner = SimulationRunner(config, output_dir="logs/my_run")
results = runner.run_learning(agent, data, n_epochs=100, n_inference_steps=20)
```

## Development Guidelines

1. **Use Real ngclearn Components**: Never mock, always use actual RateCell, GaussianErrorCell, etc.
2. **Test-Driven Development**: Write tests before implementation
3. **Type Hints**: All functions must have type annotations
4. **Docstrings**: Google style for all public APIs
5. **Logging**: Use loguru, not print statements
6. **JIT Compilation**: Use @jit for performance-critical functions
7. **Modular Design**: Clear separation of concerns

## Testing
Each module has corresponding tests:
- Unit tests: `tests/unit/test_*.py`
- Integration tests: `tests/integration/test_*.py`
- Simulation tests: `tests/simulations/test_*.py`

Run all tests: `pytest tests/ -v`

## Performance
- JIT-compiled core functions for speed
- JAX for automatic differentiation and GPU support
- Batched operations where possible
- Profiled and optimized critical paths

## Dependencies
- **jax**: Numerical computation and autodiff
- **ngclearn**: Neurobiological components
- **ngcsimlib**: Simulation infrastructure
- **loguru**: Professional logging
- **matplotlib**: Visualization

See `pyproject.toml` for complete list with versions.






