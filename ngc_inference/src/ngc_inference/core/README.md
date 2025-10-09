# Core Active Inference Algorithms

This module implements the mathematical foundations of Active Inference and the Free Energy Principle.

## Modules

### `free_energy.py`
Variational free energy computations for inference and learning.

**Key Functions:**
- `compute_free_energy(observation, prediction, prior_mean, posterior_mean, ...)`
  - Computes VFE = Accuracy + Complexity
  - Accuracy: Reconstruction error (negative log-likelihood)
  - Complexity: KL divergence between posterior and prior
  - Returns: (free_energy, components_dict)

- `compute_prediction_error(observation, prediction, precision)`
  - Precision-weighted prediction errors
  - Core signal for predictive coding
  
- `compute_expected_free_energy(predicted_obs, preferred_obs, ...)`
  - For action selection in active inference
  - Balances pragmatic value (goal achievement) and epistemic value (information gain)

- `compute_gaussian_entropy(mean, log_variance, ...)`
  - Entropy of Gaussian distributions
  - Used for epistemic value computation

- `compute_kl_divergence(mean_q, log_var_q, mean_p, log_var_p)`
  - KL(q||p) for Gaussian distributions
  - Complexity term in free energy

**Mathematical Background:**

Variational Free Energy:
```
F = -log p(o|μ) + D_KL[q(μ)||p(μ)]
  = Accuracy + Complexity
  = 0.5 * σ_obs⁻² * ||o - g(μ)||² + 0.5 * σ_prior⁻² * ||μ - μ₀||²
```

Expected Free Energy (for actions):
```
G(π) = E_q[H[p(o|s,π)]] - E_q[log p(o)] - E_q[D_KL[p(s|o,π)||q(s)]]
     = Ambiguity - Pragmatic Value - Epistemic Value
```

### `inference.py`
Base agent classes implementing Active Inference through predictive coding.

**Classes:**

**VariationalInferenceAgent**
- Base class for perception through free energy minimization
- Uses ngclearn components: RateCell, GaussianErrorCell, DenseSynapse
- Implements predictive coding architecture
- Methods:
  - `infer(observations, n_steps)`: Perform inference
  - `learn(observations, n_epochs)`: Learn generative model

**ActiveInferenceAgent** (extends VariationalInferenceAgent)
- Adds action selection via expected free energy minimization
- Methods:
  - `select_action(current_obs, preferred_obs)`: Choose optimal action
  - `_predict_observation(state, action)`: Predict outcome of action

**Architecture:**
```
Observation → Error Neuron → Recognition Weights → Hidden State
                ↑                                         ↓
              Prediction ← Generative Weights ← ← ← ← ← ← ┘
```

## Design Principles

1. **Neurobiological Plausibility**: Uses real ngclearn neurons and synapses
2. **Predictive Coding**: Top-down predictions, bottom-up errors
3. **Free Energy Minimization**: Core optimization principle
4. **Hierarchical Structure**: Supports multi-layer architectures
5. **JIT Compilation**: Performance-optimized with JAX

## Usage Examples

### Free Energy Computation
```python
from ngc_inference.core.free_energy import compute_free_energy

fe, components = compute_free_energy(
    observation=obs,
    prediction=pred,
    prior_mean=jnp.zeros_like(state),
    posterior_mean=state,
    observation_precision=1.0,
    prior_precision=1.0
)

print(f"Free Energy: {fe:.4f}")
print(f"Accuracy: {components['accuracy']:.4f}")
print(f"Complexity: {components['complexity']:.4f}")
```

### Variational Inference Agent
```python
from ngc_inference.core.inference import VariationalInferenceAgent

agent = VariationalInferenceAgent(
    name="my_agent",
    n_observations=10,
    n_hidden=20,
    learning_rate=0.01,
    precision_obs=1.0,
    precision_hidden=1.0,
    integration_steps=30
)

beliefs, metrics = agent.infer(observations, n_steps=30)
training_metrics = agent.learn(data, n_epochs=100)
```

### Active Inference with Actions
```python
from ngc_inference.core.inference import ActiveInferenceAgent

agent = ActiveInferenceAgent(
    name="active_agent",
    n_observations=10,
    n_hidden=20,
    n_actions=5,
    learning_rate=0.01
)

action, metrics = agent.select_action(
    current_observation=obs,
    preferred_observation=goal
)
```

## Testing

Unit tests in `tests/unit/test_free_energy.py` cover:
- Prediction error computation
- Free energy components (accuracy, complexity)
- Expected free energy calculation
- Gaussian entropy and KL divergence
- Precision scaling effects

Integration tests in `tests/integration/` verify:
- Agent initialization with ngclearn components
- Inference dynamics and convergence
- Free energy minimization over time
- Learning from data

## Performance

All core functions are JIT-compiled:
- `@jit` decorator for automatic compilation
- Significant speedup on repeated calls
- GPU acceleration when available
- Batched operations for efficiency

## References

1. Friston, K. (2010). "The free-energy principle: a unified brain theory?"
2. Rao & Ballard (1999). "Predictive coding in the visual cortex"
3. Friston et al. (2017). "Active inference: a process theory"
4. Ororbia & Kifer (2022). "The neural coding framework for learning generative models"






