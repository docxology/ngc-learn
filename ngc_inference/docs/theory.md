# Theoretical Background

## Active Inference and the Free Energy Principle

Active Inference is a framework for understanding perception, learning, and action based on the **Free Energy Principle** (FEP), which states that biological systems minimize their variational free energy.

### Variational Free Energy

Variational free energy \( F \) is an upper bound on surprise:

\[
F = \underbrace{-\log p(o|\mu)}\_{\text{Accuracy}} + \underbrace{D_{KL}[q(\mu)||p(\mu)]}\_{\text{Complexity}}
\]

Where:
- \( o \): observations (sensory data)
- \( \mu \): hidden states (latent causes)
- \( q(\mu) \): recognition density (approximate posterior)
- \( p(\mu) \): prior beliefs
- \( p(o|\mu) \): likelihood (generative model)

### Components

**Accuracy Term**: Measures how well the generative model predicts observations.
\[
\text{Accuracy} = \frac{1}{2\sigma_o^2} ||o - g(\mu)||^2
\]

**Complexity Term**: KL divergence between posterior and prior (regularization).
\[
\text{Complexity} = D_{KL}[q(\mu)||p(\mu)] = \frac{1}{2\sigma_\mu^2} ||\mu - \mu_0||^2
\]

### Predictive Coding

Predictive coding implements free energy minimization through **prediction errors**:

\[
\epsilon = o - g(\mu)
\]

Prediction errors drive inference (updating beliefs) and learning (updating parameters).

#### Inference Dynamics

Beliefs are updated to minimize prediction errors:

\[
\dot{\mu} = -\frac{\partial F}{\partial \mu} = \frac{1}{\sigma_o^2} g'(\mu)^T \epsilon - \frac{1}{\sigma_\mu^2}(\mu - \mu_0)
\]

#### Learning Dynamics

Generative model parameters \( \theta \) are updated to minimize expected free energy:

\[
\dot{\theta} = -\eta \frac{\partial F}{\partial \theta} = \eta \frac{1}{\sigma_o^2} \epsilon \frac{\partial g}{\partial \theta}
\]

### Hierarchical Inference

In hierarchical models, each level predicts the level below:

\[
F = \sum_{i=1}^{N} \left[ \frac{1}{2\sigma_i^2} ||\mu_{i-1} - g_i(\mu_i)||^2 + \frac{1}{2\sigma_i^2} ||\mu_i - \mu_{i,0}||^2 \right]
\]

Where:
- Level \( i \) has states \( \mu_i \)
- \( g_i(\mu_i) \) is the generative mapping from level \( i \) to \( i-1 \)
- Level 0 corresponds to observations

### Expected Free Energy

For action selection, agents minimize **expected free energy**:

\[
G(\pi) = \underbrace{E_q[H[p(o|\mu,\pi)]]}\_{\text{Ambiguity}} - \underbrace{E_q[\log p(o)]}\_{\text{Pragmatic}} - \underbrace{E_q[D_{KL}[p(\mu|o,\pi)||q(\mu)]]}\_{\text{Epistemic}}
\]

Where:
- \( \pi \): policy (action sequence)
- **Pragmatic value**: Expected log-likelihood of preferred outcomes
- **Epistemic value**: Expected information gain
- **Ambiguity**: Uncertainty about outcomes given states

## Implementation in ngclearn

### Neuronal Components

NGC Inference uses neurobiologically plausible ngclearn components:

1. **RateCell**: Continuous-valued neurons with leaky dynamics
   \[
   \tau \dot{z} = -z + \sigma(j)
   \]

2. **GaussianErrorCell**: Precision-weighted prediction errors
   \[
   \epsilon = \frac{1}{\sigma}(target - prediction)
   \]

3. **DenseSynapse**: Weighted connections with Hebbian learning
   \[
   \dot{W} = \eta \cdot \text{pre} \cdot \text{post}^T
   \]

### Network Architecture

#### Simple Prediction

```
Observation (o) ---> Error (ε) ---> Recognition (W_rec) ---> Hidden (μ)
                        ^                                        |
                        |                                        v
                    Prediction (ĥ) <--- Generation (W_gen) <----+
```

#### Hierarchical Model

```
                                    μ₃ (top level)
                                     |
                            W_gen₂   v   W_rec₂
                                    μ₂
                                     |
                            W_gen₁   v   W_rec₁
                                    μ₁
                                     |
                            W_gen₀   v   W_rec₀
                                    o (observations)
```

Each level computes prediction errors that propagate up (recognition) and predictions that propagate down (generation).

### Free Energy Computation

```python
def compute_free_energy(obs, pred, prior, posterior, σ_obs, σ_prior):
    # Accuracy: Reconstruction error
    accuracy = 0.5 * (1/σ_obs²) * ||obs - pred||²
    
    # Complexity: State divergence from prior
    complexity = 0.5 * (1/σ_prior²) * ||posterior - prior||²
    
    return accuracy + complexity
```

### Inference Process

1. **Initialize** states to priors
2. **Forward pass**: Compute predictions top-down
3. **Compute errors**: Compare predictions to targets
4. **Backward pass**: Update states to minimize errors
5. **Repeat** until convergence (free energy minimum)

### Learning Process

1. Perform inference to convergence
2. Compute parameter gradients from prediction errors
3. Update generative and recognition weights
4. Repeat for multiple epochs

## Key Properties

### Bayesian Inference
Free energy minimization performs approximate Bayesian inference:
- Posterior ≈ Recognition model output
- Prior beliefs regularize inference
- Precision = inverse variance (confidence)

### Hierarchical Abstraction
Higher levels encode:
- Slower dynamics
- More abstract features
- Context and structure

Lower levels encode:
- Faster dynamics  
- Detailed features
- Sensory specifics

### Active Inference
Extending to action:
- Perception: Minimize free energy by updating beliefs
- Action: Minimize expected free energy by changing observations
- Unifies perception and action under single principle

## References

1. Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*.

2. Rao, R. P., & Ballard, D. H. (1999). "Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects." *Nature Neuroscience*.

3. Buckley, C. L., et al. (2017). "The free energy principle for action and perception: A mathematical review." *Journal of Mathematical Psychology*.

4. Friston, K., et al. (2017). "Active inference: a process theory." *Neural Computation*.

5. Ororbia, A., & Kifer, D. (2022). "The neural coding framework for learning generative models." *Nature Communications*.

## Active Inference: Complete Formulation

Active Inference extends variational free energy minimization to action selection, providing a unified framework for perception, learning, and behavior.

### Variational Free Energy (VFE) for Perception

Variational free energy minimization performs approximate Bayesian inference:

\[
F = \underbrace{-\log p(o|\mu)}\_{\text{Accuracy}} + \underbrace{D_{KL}[q(\mu)||p(\mu)]}\_{\text{Complexity}}
\]

**Accuracy Term**: Measures how well generative model predictions match observations:
\[
\text{Accuracy} = \frac{1}{2\sigma_o^2} ||o - g(\mu)||^2
\]

**Complexity Term**: Regularizes beliefs toward prior expectations:
\[
\text{Complexity} = \frac{1}{2\sigma_s^2} ||\mu - \mu_0||^2
\]

### Gradient Descent on Beliefs

Beliefs are updated via gradient descent on free energy:

\[
\dot{\mu} = -\eta \frac{\partial F}{\partial \mu} = -\eta \left[ \frac{1}{\sigma_o^2} W_{gen}^T (g(\mu) - o) + \frac{1}{\sigma_s^2} (\mu - \mu_0) \right]
\]

Where:
- \( g(\mu) = W_{gen} \mu \) is the generative mapping
- \( W_{gen} \) are generative weights
- \( \eta \) is the learning rate for belief updates

### Expected Free Energy (EFE) for Action Selection

Actions are selected to minimize expected free energy over policies:

\[
G(\pi) = \underbrace{E_q[H[p(o|s,\pi)]]}\_{\text{Ambiguity}} - \underbrace{E_q[\log p(o)]}\_{\text{Pragmatic}} - \underbrace{E_q[D_{KL}[p(s|o,\pi)||q(s)]]}\_{\text{Epistemic}}
\]

**Pragmatic Value**: Expected log-likelihood of preferred outcomes:
\[
\text{Pragmatic} = E_q[\log p(o^*|s,\pi)] = -\frac{1}{2\sigma_o^2} ||E[g(\mu)|o^*] - o^*||^2
\]

**Epistemic Value**: Expected information gain about states:
\[
\text{Epistemic} = E_q[D_{KL}[p(s|o,\pi)||q(s)]] = H[p(s|o,\pi)] - H[q(s)]
\]

**Ambiguity**: Expected uncertainty about observations:
\[
\text{Ambiguity} = E_q[H[p(o|s,\pi)]] = \frac{1}{2} \log(2\pi e \sigma_o^2)
\]

### Policy Posterior

Actions are sampled from the policy posterior:

\[
q(\pi) = \frac{\exp(-G(\pi)/\gamma)}{\sum_{\pi'} \exp(-G(\pi')/\gamma)}
\]

Where \( \gamma \) is the policy temperature controlling exploration vs exploitation.

### Transition Models p(s'|s,a)

State transitions are modeled to predict outcomes of actions:

**Discrete Transitions**: Transition matrices T[a] where T[a][i,j] = p(s'=j|s=i,a)

**Continuous Transitions**: Neural network f(s,a) where s' = f(s,a) + ε, ε ~ N(0,Σ)

### Active Inference Loop

1. **Perceive**: Minimize VFE to infer current state beliefs μ
2. **Plan**: Evaluate EFE for all possible actions/policies
3. **Act**: Sample action from policy posterior q(π)
4. **Learn**: Update transition and generative models from experience

### Implementation in NGC Inference

The framework implements:

**Core Components:**
- `compute_state_gradients()`: ∂F/∂μ for belief updates
- `compute_policy_posterior()`: q(π) = softmax(-G/γ)
- `TransitionModel` classes: p(s'|s,a) for discrete and continuous actions

**Agent Classes:**
- `ActiveInferenceAgent`: Complete implementation with VFE + EFE
- `SimplePredictionAgent`: Single-layer predictive coding (fixed)
- `HierarchicalInferenceAgent`: Multi-layer hierarchical inference (fixed)

**Key Features:**
- Proper VFE minimization via gradient descent (fixes failing tests)
- Full EFE computation with pragmatic + epistemic values
- Policy posterior sampling for stochastic action selection
- Learnable transition models from experience
- Support for discrete and continuous action spaces
- Neurobiologically plausible ngclearn integration

### Mathematical Derivations

**Belief Gradient:**
\[
\frac{\partial F}{\partial \mu} = \frac{\partial}{\partial \mu} \left[ \frac{1}{2\sigma_o^2} ||o - W_{gen}\mu||^2 + \frac{1}{2\sigma_s^2} ||\mu - \mu_0||^2 \right]
\]
\[
= -\frac{1}{\sigma_o^2} W_{gen}^T (o - W_{gen}\mu) + \frac{1}{\sigma_s^2} (\mu - \mu_0)
\]

**Policy Posterior:**
\[
q(\pi) \propto \exp\left(-\frac{G(\pi)}{\gamma}\right) = \exp\left(-\frac{1}{\gamma} (-\text{Pragmatic} - \text{Epistemic})\right)
\]

**Information Gain:**
\[
I[s;o] = H[p(s)] - H[p(s|o)] = \frac{1}{2} \log\left(\frac{\sigma_s^2}{\sigma_{s|o}^2}\right)
\]

### Neurobiological Interpretation

**Predictive Coding**: Top-down predictions, bottom-up prediction errors
**Free Energy Principle**: All behavior minimizes variational free energy
**Active Inference**: Perception minimizes VFE, action minimizes EFE
**Bayesian Brain**: Beliefs represent posterior distributions over causes

### References

1. Friston, K. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*.
2. Friston, K., et al. (2017). "Active inference: a process theory." *Neural Computation*.
3. Parr, T., & Friston, K. J. (2018). "The active inference approach to ecological perception: general information dynamics for natural and artificial embodied systems." *Frontiers in Robotics and AI*.
4. Ororbia, A., & Kifer, D. (2022). "The neural coding framework for learning generative models." *Nature Communications*.




