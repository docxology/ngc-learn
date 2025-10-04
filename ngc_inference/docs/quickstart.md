# Quick Start Guide

This guide will help you get started with NGC Inference for Active Inference simulations.

## Installation

### Using uv (Recommended)

```bash
cd ngc_inference
./scripts/setup_environment.sh
```

Or manually:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,docs]"
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,docs]"
```

## Verify Installation

```bash
python scripts/verify_installation.py
```

## First Example: Simple Prediction

```python
import jax.numpy as jnp
from jax import random
from ngc_inference.simulations.simple_prediction import SimplePredictionAgent

# Create agent
agent = SimplePredictionAgent(
    n_observations=10,
    n_hidden=20,
    learning_rate=0.01
)

# Generate some data
key = random.PRNGKey(42)
data = random.normal(key, (50, 10))

# Learn from data
results = agent.learn(data, n_epochs=50)
print(f"Final loss: {results['final_loss']:.4f}")

# Perform inference
observation = data[0:1]
beliefs, metrics = agent.infer(observation, n_steps=30)
print(f"Free energy: {metrics['free_energy']:.4f}")
```

## Run Pre-built Examples

### Simple Prediction
```bash
python scripts/run_simple_example.py
```

### Hierarchical Inference
```bash
python scripts/run_hierarchical_example.py
```

## Core Concepts

### 1. Free Energy Minimization

Active Inference agents minimize variational free energy:

```python
from ngc_inference.core.free_energy import compute_free_energy

fe, components = compute_free_energy(
    observation,        # Sensory data
    prediction,         # Model prediction
    prior_mean,        # Prior belief
    posterior_mean,    # Inferred belief
    observation_precision=1.0,
    prior_precision=1.0
)
```

Free Energy = Accuracy + Complexity
- **Accuracy**: How well predictions match observations
- **Complexity**: Divergence of beliefs from priors

### 2. Agents

#### Simple Prediction Agent
Single-layer predictive coding for learning sensory models.

```python
from ngc_inference.simulations.simple_prediction import SimplePredictionAgent

agent = SimplePredictionAgent(
    n_observations=10,
    n_hidden=20,
    learning_rate=0.01,
    precision=1.0
)
```

#### Hierarchical Inference Agent
Multi-layer predictive hierarchy for deep inference.

```python
from ngc_inference.simulations.hierarchical_inference import HierarchicalInferenceAgent

agent = HierarchicalInferenceAgent(
    layer_sizes=[10, 20, 15, 10],  # observation, hidden1, hidden2, hidden3
    learning_rate=0.005,
    precisions=[1.0, 1.0, 1.0, 1.0]
)
```

### 3. Orchestration

Use orchestrators to manage simulation workflows:

```python
from ngc_inference.orchestrators.simulation_runner import SimulationRunner

config = {
    "agent": {"n_observations": 10, "n_hidden": 20},
    "training": {"n_epochs": 50, "n_inference_steps": 20}
}

runner = SimulationRunner(config, output_dir="logs/my_experiment")

# Run learning
results = runner.run_learning(
    agent,
    observations,
    n_epochs=50,
    n_inference_steps=20,
    save_results=True
)
```

### 4. Configuration

Use YAML configuration files:

```yaml
# config.yaml
simulation:
  name: "my_experiment"
  seed: 42

agent:
  n_observations: 10
  n_hidden: 20
  learning_rate: 0.01

training:
  n_epochs: 100
  n_inference_steps: 20
```

Load and use:

```python
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

runner = SimulationRunner(config)
```

### 5. Visualization

```python
from ngc_inference.utils.visualization import plot_free_energy, plot_beliefs

# Plot free energy trajectory
plot_free_energy(
    metrics["free_energy_trajectory"],
    save_path="free_energy.png"
)

# Plot beliefs and observations
plot_beliefs(
    beliefs,
    observations,
    predictions,
    save_path="beliefs.png"
)
```

## Next Steps

- Read the [User Guide](user_guide.md) for detailed documentation
- Explore [Examples](examples.md) for more use cases
- Check [API Reference](api.md) for complete API documentation
- See [Theory](theory.md) for mathematical background

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
pytest tests/unit/ -m unit -v              # Unit tests
pytest tests/integration/ -m integration -v  # Integration tests
pytest tests/simulations/ -m simulation -v   # Simulation tests
```

## Getting Help

- Check the [FAQ](faq.md)
- See [Troubleshooting](troubleshooting.md)
- Review [ngclearn documentation](https://ngc-learn.readthedocs.io/)




