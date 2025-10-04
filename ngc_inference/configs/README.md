# Configuration Files

YAML configuration files for NGC Inference simulations and experiments.

## Files

### `simple_prediction.yaml`
Configuration for SimplePredictionAgent simulations.

**Sections**:
- **simulation**: Name, type, seed
- **agent**: Architecture and hyperparameters
- **training**: Epochs and inference steps
- **inference**: Inference-only parameters
- **data**: Data generation settings
- **logging**: Log level and file
- **output**: Results saving options

**Usage**:
```python
import yaml
with open("configs/simple_prediction.yaml") as f:
    config = yaml.safe_load(f)

agent = SimplePredictionAgent(**config["agent"])
runner = SimulationRunner(config)
```

### `hierarchical_inference.yaml`
Configuration for HierarchicalInferenceAgent simulations.

**Key Differences from Simple**:
- `layer_sizes`: List instead of n_hidden
- `precisions`: Per-level precision values
- Typically more epochs and inference steps

### `experiment_template.yaml`
Template for parameter sweep experiments.

**Structure**:
- **base_config**: Shared configuration
- **parameter_grid**: Parameters to sweep
- **execution**: Parallelization settings
- **analysis**: Metric for optimization

**Usage**:
```python
manager = ExperimentManager(
    base_config=config["base_config"],
    experiment_name=config["experiment"]["name"]
)

configs = manager.create_parameter_grid(config["parameter_grid"])
results = manager.run_experiment(configs, run_fn)
```

## Configuration Schema

### Simulation Section
```yaml
simulation:
  name: "experiment_name"           # Unique identifier
  type: "SimplePredictionAgent"     # Agent class name
  seed: 42                           # Random seed for reproducibility
```

### Agent Section (Simple)
```yaml
agent:
  n_observations: 10                # Observation dimensionality
  n_hidden: 20                      # Hidden state dimensionality
  batch_size: 1                     # Batch size
  learning_rate: 0.01               # Synaptic learning rate
  precision: 1.0                    # Observation precision
  tau: 10.0                         # Time constant (ms)
  dt: 1.0                           # Integration step (ms)
```

### Agent Section (Hierarchical)
```yaml
agent:
  layer_sizes: [10, 20, 15, 10]     # [obs, h1, h2, h3]
  batch_size: 1
  learning_rate: 0.005              # Often lower for stability
  precisions: [1.0, 1.0, 1.0, 1.0]  # Per-level precisions
  tau: 10.0
  dt: 1.0
```

### Training Section
```yaml
training:
  n_epochs: 100                     # Training epochs
  n_inference_steps: 20             # Inference iterations per sample
  verbose: true                     # Print progress
```

### Inference Section
```yaml
inference:
  n_steps: 30                       # Inference iterations (test time)
```

### Data Section
```yaml
data:
  n_samples: 50                     # Number of training samples
  noise_level: 0.1                  # Gaussian noise std
  data_type: "sinusoidal"           # Data generation method
  # Options: sinusoidal, random, structured, hierarchical
```

### Logging Section
```yaml
logging:
  level: "INFO"                     # Log level (DEBUG, INFO, WARNING, ERROR)
  log_file: "logs/experiment.log"   # Log file path (null for console only)
  save_plots: true                  # Generate visualization plots
```

### Output Section
```yaml
output:
  save_results: true                # Save metrics and outputs
  output_dir: null                  # Auto-generate if null
```

## Parameter Grid Format

For experiments:
```yaml
parameter_grid:
  agent.learning_rate: [0.001, 0.01, 0.1]      # Dot notation for nested
  agent.precision: [0.5, 1.0, 2.0]
  training.n_epochs: [50, 100, 150]
```

Generates all combinations (Cartesian product): 3 × 3 × 3 = 27 configurations

## Creating Custom Configurations

### Step 1: Copy Template
```bash
cp configs/simple_prediction.yaml configs/my_experiment.yaml
```

### Step 2: Edit Parameters
```yaml
simulation:
  name: "my_custom_experiment"
  seed: 123

agent:
  n_observations: 20
  n_hidden: 50
  learning_rate: 0.005
```

### Step 3: Use in Code
```python
with open("configs/my_experiment.yaml") as f:
    config = yaml.safe_load(f)

agent = SimplePredictionAgent(**config["agent"])
```

## Best Practices

1. **Version Control**: Commit configuration files with code
2. **Documentation**: Add comments explaining choices
3. **Reproducibility**: Always set seed
4. **Validation**: Test configurations before long runs
5. **Naming**: Use descriptive experiment names
6. **Organization**: Group related experiments in folders

## Parameter Selection Guidelines

### Learning Rates
- **Simple Agent**: 0.01 - 0.05
- **Hierarchical Agent**: 0.001 - 0.01
- **Rule**: Lower for deeper/larger networks

### Inference Steps
- **Simple Agent**: 20-30 steps
- **Hierarchical Agent**: 30-50 steps
- **Rule**: More steps for deeper hierarchies

### Precision
- **High Precision (2.0+)**: When confident in observations
- **Medium Precision (1.0)**: Standard case
- **Low Precision (0.5)**: Noisy/uncertain observations

### Layer Sizes (Hierarchical)
- **Bottleneck**: Decreasing then increasing [10, 8, 6, 8, 10]
- **Expanding**: Increasing [10, 20, 30, 40]
- **Balanced**: Similar sizes [10, 15, 12, 10]

## Validation

Before running experiments, validate configuration:

```python
def validate_config(config):
    """Validate configuration file."""
    assert "agent" in config
    assert "training" in config
    
    if config.get("simulation", {}).get("type") == "SimplePredictionAgent":
        assert "n_observations" in config["agent"]
        assert "n_hidden" in config["agent"]
    
    elif config.get("simulation", {}).get("type") == "HierarchicalInferenceAgent":
        assert "layer_sizes" in config["agent"]
        assert len(config["agent"]["layer_sizes"]) >= 2
    
    return True
```

## Examples Library

Common configurations:

**Quick Test**:
```yaml
training:
  n_epochs: 10
  n_inference_steps: 10
```

**Production Run**:
```yaml
training:
  n_epochs: 200
  n_inference_steps: 50
```

**Large Scale**:
```yaml
agent:
  n_observations: 100
  n_hidden: 200
training:
  n_epochs: 500
```

## Troubleshooting

**Issue**: Simulation crashes
- **Check**: Valid parameter ranges
- **Verify**: Required fields present
- **Test**: Load config before running

**Issue**: Poor performance
- **Adjust**: Learning rate (often too high)
- **Increase**: Inference steps
- **Modify**: Precision values

**Issue**: Slow convergence
- **Increase**: Learning rate
- **Add**: More epochs
- **Check**: Network architecture

## Configuration Management

Track experiments:
```bash
configs/
├── simple_prediction.yaml       # Base template
├── hierarchical_inference.yaml  # Hierarchical template
├── experiment_template.yaml     # Parameter sweep template
└── experiments/                 # Custom experiments
    ├── exp001_baseline.yaml
    ├── exp002_high_lr.yaml
    └── exp003_deep_hierarchy.yaml
```

## Integration with Code

Configurations integrate seamlessly:

```python
# Load
with open("configs/simple_prediction.yaml") as f:
    config = yaml.safe_load(f)

# Create agent from config
agent = SimplePredictionAgent(**config["agent"])

# Create runner from config
runner = SimulationRunner(config)

# Run with config parameters
results = runner.run_learning(
    agent,
    data,
    n_epochs=config["training"]["n_epochs"],
    n_inference_steps=config["training"]["n_inference_steps"]
)
```

## Version Compatibility

Configuration format is stable across versions:
- v0.1.x: Current format
- Future: Backward compatible or migration tools provided

Always specify version in config:
```yaml
version: "0.1.0"
```




