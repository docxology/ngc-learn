# Active Inference Agents

This document describes the agent implementations for Active Inference simulations using ngclearn.

## Agent Hierarchy

```
VariationalInferenceAgent (base class in core/inference.py)
    ↓
ActiveInferenceAgent (extends with action selection)
    ↓
SimplePredictionAgent (single-layer concrete implementation)
    ↓
HierarchicalInferenceAgent (multi-layer concrete implementation)
```

## SimplePredictionAgent

**Location**: `simple_prediction.py`

**Purpose**: Single-layer predictive coding for learning sensory generative models.

### Architecture

```
Observation (o)
    ↓
Error Neuron (ε = o - ô)
    ↓
Recognition Weights (W_rec)
    ↓
Hidden State (μ)
    ↓
Generative Weights (W_gen)
    ↓
Prediction (ô)
    ↑_____↑ (error signal feeds back)
```

### Components
- **z_hidden**: RateCell (n_hidden units) - latent representation
- **z_pred**: RateCell (n_observations units) - observation prediction  
- **e_obs**: GaussianErrorCell - computes precision-weighted errors
- **W_gen**: DenseSynapse (n_hidden → n_observations) - generative/top-down
- **W_rec**: DenseSynapse (n_observations → n_hidden) - recognition/bottom-up

### Parameters
- `n_observations`: Observation dimensionality
- `n_hidden`: Hidden state dimensionality
- `learning_rate`: Synaptic learning rate (default: 0.01)
- `precision`: Observation precision/inverse variance (default: 1.0)
- `tau`: Time constant for neuronal dynamics (default: 10.0)
- `dt`: Integration time step (default: 1.0)

### Methods

**`infer(observation, n_steps=20)`**
- Performs inference by minimizing prediction errors
- Iteratively updates hidden states
- Returns: (beliefs, metrics)
- Metrics include: free_energy, free_energy_trajectory, prediction_error

**`learn(observations, n_epochs=100, n_inference_steps=20)`**
- Learns generative model from data
- Performs inference then updates weights
- Returns: training metrics with losses and learned weights

**`predict(hidden_states)`**
- Generates observations from hidden states
- Uses learned generative weights
- Returns: predicted observations

### Use Cases
1. **Feature Learning**: Discover latent structure in sensory data
2. **Sensory Prediction**: Learn to predict upcoming observations
3. **Denoising**: Infer clean signal from noisy observations
4. **Dimensionality Reduction**: Compress high-D observations to low-D representations

### Example
```python
agent = SimplePredictionAgent(
    n_observations=10,
    n_hidden=20,
    learning_rate=0.01,
    precision=1.0
)

# Inference on single observation
beliefs, metrics = agent.infer(observation, n_steps=30)
print(f"Free Energy: {metrics['free_energy']:.4f}")

# Learn from dataset
results = agent.learn(data, n_epochs=100, n_inference_steps=20)
print(f"Final Loss: {results['final_loss']:.4f}")

# Generate prediction
prediction = agent.predict(beliefs)
```

## HierarchicalInferenceAgent

**Location**: `hierarchical_inference.py`

**Purpose**: Multi-layer hierarchical predictive coding for deep inference.

### Architecture

```
              μ₃ (top level - abstract)
               ↓ W_gen₂  ↑ W_rec₂
              μ₂ (mid level)
               ↓ W_gen₁  ↑ W_rec₁  
              μ₁ (low level)
               ↓ W_gen₀  ↑ W_rec₀
              o (observations - concrete)
```

Each level:
- Predicts the level below (top-down)
- Computes prediction errors
- Sends errors upward (bottom-up)

### Components (per layer)
- **States**: RateCell neurons at each hidden level
- **Predictions**: RateCell neurons predicting each level
- **Errors**: GaussianErrorCell at each level
- **Generative Synapses**: Top-down prediction weights
- **Recognition Synapses**: Bottom-up error weights

### Parameters
- `layer_sizes`: List [obs_dim, hidden1_dim, hidden2_dim, ...] 
- `learning_rate`: Synaptic learning rate (default: 0.01)
- `precisions`: List of precisions for each level (default: all 1.0)
- `tau`: Time constant (default: 10.0)
- `dt`: Integration time step (default: 1.0)

### Methods

**`infer(observation, n_steps=30)`**
- Performs hierarchical inference
- Minimizes prediction errors at all levels simultaneously
- Returns: (beliefs_list, metrics)
- beliefs_list: List of inferred states at each level

**`learn(observations, n_epochs=100, n_inference_steps=30)`**
- Learns hierarchical generative model
- Updates all generative and recognition weights
- Returns: training metrics with losses and all learned weights

**`generate(top_level_state)`**
- Top-down generation from abstract representation
- Sequentially generates predictions at each level
- Returns: list of predictions at all levels

### Properties
- **Hierarchical Abstraction**: Higher levels encode more abstract features
- **Temporal Hierarchy**: Higher levels typically have slower dynamics
- **Contextual Modulation**: Top-down predictions provide context
- **Error Propagation**: Errors propagate up, predictions down

### Use Cases
1. **Deep Feature Learning**: Multi-level representations
2. **Hierarchical Prediction**: Context-aware forecasting
3. **Abstract Reasoning**: High-level concept learning
4. **Generative Modeling**: Hierarchical data generation

### Example
```python
agent = HierarchicalInferenceAgent(
    layer_sizes=[10, 20, 15, 10],  # 3 hidden layers
    learning_rate=0.005,
    precisions=[1.0, 1.0, 1.0, 1.0]
)

# Hierarchical inference
beliefs_list, metrics = agent.infer(observation, n_steps=50)
print(f"Hierarchical FE: {metrics['free_energy']:.4f}")
for i, beliefs in enumerate(beliefs_list):
    print(f"Level {i+1} shape: {beliefs.shape}")

# Learn hierarchical model
results = agent.learn(data, n_epochs=150, n_inference_steps=30)

# Top-down generation
top_state = jnp.random.normal(key, (1, 10))
generated = agent.generate(top_state)
observation_prediction = generated[0]  # Lowest level prediction
```

## Comparison

| Feature | Simple | Hierarchical |
|---------|--------|--------------|
| Layers | 1 | Multiple |
| Complexity | Low | High |
| Abstraction | Single-level | Multi-level |
| Use Case | Basic prediction | Deep inference |
| Inference Steps | 20-30 | 30-50 |
| Learning Rate | 0.01 | 0.005-0.01 |
| Parameters | ~n_obs × n_hidden | Sum over all layers |

## Design Principles

1. **Real ngclearn Components**: All agents use actual RateCell, GaussianErrorCell, DenseSynapse
2. **Predictive Coding**: Top-down predictions, bottom-up errors
3. **Free Energy Minimization**: Core optimization across all levels
4. **Modular Architecture**: Easy to extend and customize
5. **Production Ready**: Tested, documented, logged

## Agent Selection Guide

**Use SimplePredictionAgent when:**
- Learning basic sensory predictions
- Single-level feature extraction
- Fast prototyping and testing
- Limited computational resources

**Use HierarchicalInferenceAgent when:**
- Deep representations needed
- Multi-scale structure in data
- Abstract concept learning
- Hierarchical generative modeling

## Future Agents

Planned extensions:
1. **TemporalInferenceAgent**: Sequential prediction over time
2. **MultiModalAgent**: Vision + audio + proprioception
3. **ContinualLearningAgent**: Online adaptation without forgetting
4. **MetaLearningAgent**: Learning to learn quickly
5. **NeuromorphicAgent**: Deployment on neuromorphic hardware (Lava)

## Testing

All agents have comprehensive tests:
- **Unit Tests**: Component initialization
- **Integration Tests**: Inference and learning workflows  
- **Simulation Tests**: End-to-end with orchestrators
- **Verification Tests**: Real ngclearn compatibility

Run tests: `pytest tests/ -v -m integration`

## Performance Benchmarks

**SimplePredictionAgent** (n_obs=10, n_hidden=20):
- Inference (30 steps): ~50ms
- Training (100 epochs, 50 samples): ~30s

**HierarchicalInferenceAgent** ([10,20,15,10]):
- Inference (50 steps): ~150ms  
- Training (150 epochs, 50 samples): ~90s

(MacBook Pro M1, CPU only)

## References

- Ororbia & Kifer (2022): ngclearn framework
- Friston (2010): Free Energy Principle
- Rao & Ballard (1999): Predictive Coding
- Whittington & Bogacz (2017): Hierarchical Predictive Coding






