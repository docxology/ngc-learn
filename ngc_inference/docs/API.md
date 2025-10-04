# API Reference

## Core Modules

### Free Energy (`ngc_inference.core.free_energy`)

#### `compute_free_energy`
```python
compute_free_energy(
    observation: Array,
    prediction: Array,
    prior_mean: Array,
    posterior_mean: Array,
    observation_precision: float = 1.0,
    prior_precision: float = 1.0
) -> Tuple[Array, Dict]
```

Compute variational free energy.

**Parameters:**
- `observation`: Observed data
- `prediction`: Model prediction
- `prior_mean`: Prior belief mean
- `posterior_mean`: Posterior belief mean
- `observation_precision`: Observation precision (inverse variance)
- `prior_precision`: Prior precision

**Returns:**
- `free_energy`: Scalar free energy value
- `components`: Dict with accuracy and complexity terms

#### `compute_prediction_error`
```python
compute_prediction_error(
    observation: Array,
    prediction: Array,
    precision: float = 1.0
) -> Array
```

Compute precision-weighted prediction error.

#### `compute_expected_free_energy`
```python
compute_expected_free_energy(
    predicted_observation: Array,
    preferred_observation: Array,
    state_entropy: Optional[Array] = None,
    observation_precision: float = 1.0,
    entropy_weight: float = 1.0
) -> Tuple[Array, Dict]
```

Compute expected free energy for action selection.

---

## Simulations

### SimplePredictionAgent

```python
class SimplePredictionAgent(
    n_observations: int,
    n_hidden: int,
    batch_size: int = 1,
    learning_rate: float = 0.01,
    precision: float = 1.0,
    tau: float = 10.0,
    dt: float = 1.0,
    seed: int = 42
)
```

Single-layer predictive coding agent.

**Methods:**

#### `infer`
```python
infer(observation: Array, n_steps: int = 20) -> Tuple[Array, Dict]
```
Perform inference on observation.

**Returns:**
- `beliefs`: Inferred hidden states
- `metrics`: Inference metrics including free energy

#### `learn`
```python
learn(
    observations: Array,
    n_epochs: int = 100,
    n_inference_steps: int = 20,
    verbose: bool = True
) -> Dict
```
Learn generative model from observations.

**Returns:**
- Training metrics including losses and learned weights

#### `predict`
```python
predict(hidden_states: Array) -> Array
```
Generate predictions from hidden states.

---

### HierarchicalInferenceAgent

```python
class HierarchicalInferenceAgent(
    layer_sizes: List[int],
    batch_size: int = 1,
    learning_rate: float = 0.01,
    precisions: Optional[List[float]] = None,
    tau: float = 10.0,
    dt: float = 1.0,
    seed: int = 42
)
```

Multi-layer hierarchical predictive coding agent.

**Parameters:**
- `layer_sizes`: List of dimensions [obs, hidden1, hidden2, ...]

**Methods:**

#### `infer`
```python
infer(observation: Array, n_steps: int = 30) -> Tuple[List[Array], Dict]
```
Perform hierarchical inference.

**Returns:**
- `beliefs`: List of inferred states at each level
- `metrics`: Hierarchical inference metrics

#### `learn`
```python
learn(
    observations: Array,
    n_epochs: int = 100,
    n_inference_steps: int = 30,
    verbose: bool = True
) -> Dict
```
Learn hierarchical generative model.

#### `generate`
```python
generate(top_level_state: Array) -> List[Array]
```
Generate observations from top-level state (top-down).

---

## Orchestrators

### SimulationRunner

```python
class SimulationRunner(config: Dict, output_dir: Optional[str] = None)
```

Orchestrates simulation execution with configuration management.

**Methods:**

#### `run_inference`
```python
run_inference(
    agent: Any,
    observations: Array,
    n_steps: int,
    save_results: bool = True
) -> Dict
```
Run inference with an agent.

#### `run_learning`
```python
run_learning(
    agent: Any,
    observations: Array,
    n_epochs: int,
    n_inference_steps: int,
    save_results: bool = True,
    callback: Optional[Callable] = None
) -> Dict
```
Run learning with an agent.

---

### ExperimentManager

```python
class ExperimentManager(
    base_config: Dict,
    experiment_name: str,
    output_dir: Optional[str] = None
)
```

Manages multiple simulations with parameter sweeps.

**Methods:**

#### `create_parameter_grid`
```python
create_parameter_grid(param_grid: Dict[str, List]) -> List[Dict]
```
Create configurations from parameter grid.

#### `run_experiment`
```python
run_experiment(
    configs: List[Dict],
    run_function: Callable,
    parallel: bool = False,
    max_workers: Optional[int] = None
) -> List[Dict]
```
Run experiment with multiple configurations.

#### `analyze_results`
```python
analyze_results(
    results: List[Dict],
    metric_key: str = "final_loss"
) -> Dict
```
Analyze experiment results.

---

## Utilities

### Metrics (`ngc_inference.utils.metrics`)

#### `InferenceMetrics`
Container for tracking inference metrics over time.

```python
class InferenceMetrics:
    def add(fe: float, pe: float, complexity: float, accuracy: float)
    def get_summary() -> Dict
    def reset()
```

#### `compute_metrics`
```python
compute_metrics(
    observations: Array,
    predictions: Array,
    beliefs: Array,
    priors: Array
) -> Dict
```

Compute comprehensive inference metrics.

#### `compute_rmse`, `compute_mae`, `compute_r2_score`
Standard regression metrics.

---

### Visualization (`ngc_inference.utils.visualization`)

#### `plot_free_energy`
```python
plot_free_energy(
    free_energy_trajectory: List[float],
    title: str = "Free Energy Minimization",
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure
```

Plot free energy trajectory.

#### `plot_beliefs`
```python
plot_beliefs(
    beliefs: Array,
    observations: Optional[Array] = None,
    predictions: Optional[Array] = None,
    title: str = "Beliefs and Observations",
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure
```

Plot beliefs, observations, and predictions.

#### `plot_metrics_comparison`
```python
plot_metrics_comparison(
    metrics_dict: Dict,
    title: str = "Metrics Comparison",
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure
```

Plot multiple metrics for comparison.

---

### Logging (`ngc_inference.utils.logging_config`)

#### `setup_logging`
```python
setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    format_string: Optional[str] = None
)
```

Configure logging for the application.

#### `get_logger`
```python
get_logger(name: str) -> Logger
```

Get a logger instance.

---

## Configuration Format

### YAML Configuration

```yaml
simulation:
  name: "experiment_name"
  type: "SimplePredictionAgent"  # or "HierarchicalInferenceAgent"
  seed: 42

agent:
  n_observations: 10
  n_hidden: 20
  learning_rate: 0.01
  precision: 1.0
  # For hierarchical:
  # layer_sizes: [10, 20, 15, 10]
  # precisions: [1.0, 1.0, 1.0, 1.0]

training:
  n_epochs: 100
  n_inference_steps: 20
  verbose: true

inference:
  n_steps: 30

logging:
  level: "INFO"
  log_file: "logs/experiment.log"
  save_plots: true

output:
  save_results: true
  output_dir: null  # Auto-generate if null
```

### Parameter Grid (for experiments)

```yaml
experiment:
  name: "parameter_sweep"

parameter_grid:
  agent.learning_rate: [0.001, 0.01, 0.1]
  agent.precision: [0.5, 1.0, 2.0]

execution:
  parallel: false
  max_workers: 4
```

---

## Type Aliases

```python
Array = jax.numpy.ndarray
Dict = Dict[str, Any]
```




