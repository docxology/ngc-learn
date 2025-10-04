"""
Free energy computation for Active Inference.

Implements variational free energy calculations following the Free Energy Principle,
using ngclearn components for neurobiologically plausible inference.
"""

from typing import Tuple, Optional
import jax.numpy as jnp
from jax import jit
from functools import partial


@jit
def compute_prediction_error(
    observation: jnp.ndarray,
    prediction: jnp.ndarray,
    precision: float = 1.0
) -> jnp.ndarray:
    """
    Compute prediction error weighted by precision.
    
    Args:
        observation: Observed data (sensory input)
        prediction: Predicted observation (generative model output)
        precision: Precision (inverse variance) of observation noise
        
    Returns:
        Prediction error signal
    """
    error = observation - prediction
    weighted_error = precision * error
    return weighted_error


@jit
def compute_free_energy(
    observation: jnp.ndarray,
    prediction: jnp.ndarray,
    prior_mean: jnp.ndarray,
    posterior_mean: jnp.ndarray,
    observation_precision: float = 1.0,
    prior_precision: float = 1.0
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute variational free energy (negative ELBO).
    
    Free Energy = Accuracy - Complexity
    where:
    - Accuracy: How well predictions match observations (reconstruction error)
    - Complexity: KL divergence between posterior and prior
    
    Args:
        observation: Observed data
        prediction: Predicted observation from generative model
        prior_mean: Prior belief mean
        posterior_mean: Posterior belief mean (recognition model output)
        observation_precision: Precision of observation likelihood
        prior_precision: Precision of prior
        
    Returns:
        total_free_energy: Scalar free energy value
        components: Dictionary with accuracy and complexity terms
    """
    # Accuracy term: negative log-likelihood of observations
    prediction_error = observation - prediction
    accuracy = 0.5 * observation_precision * jnp.sum(jnp.square(prediction_error))
    
    # Complexity term: KL divergence between posterior and prior
    state_deviation = posterior_mean - prior_mean
    complexity = 0.5 * prior_precision * jnp.sum(jnp.square(state_deviation))
    
    # Total variational free energy
    free_energy = accuracy + complexity
    
    components = {
        "accuracy": accuracy,
        "complexity": complexity,
        "prediction_error": jnp.sqrt(jnp.mean(jnp.square(prediction_error))),
        "state_divergence": jnp.sqrt(jnp.mean(jnp.square(state_deviation)))
    }
    
    return free_energy, components


@jit
def compute_expected_free_energy(
    predicted_observation: jnp.ndarray,
    preferred_observation: jnp.ndarray,
    state_entropy: Optional[jnp.ndarray] = None,
    observation_precision: float = 1.0,
    entropy_weight: float = 1.0
) -> Tuple[jnp.ndarray, dict]:
    """
    Compute expected free energy for action selection (active inference).
    
    G = Ambiguity - Pragmatic_Value - Epistemic_Value
    
    Args:
        predicted_observation: Expected observation under policy
        preferred_observation: Goal/preferred observation
        state_entropy: Entropy of state beliefs (epistemic value)
        observation_precision: Precision of observations
        entropy_weight: Weight for epistemic value (information gain)
        
    Returns:
        expected_free_energy: Scalar EFE value
        components: Dictionary with component terms
    """
    # Pragmatic value: Expected log-likelihood of preferred outcomes
    goal_error = predicted_observation - preferred_observation
    pragmatic_value = -0.5 * observation_precision * jnp.sum(jnp.square(goal_error))
    
    # Epistemic value: Information gain (state entropy reduction)
    epistemic_value = 0.0
    if state_entropy is not None:
        epistemic_value = -entropy_weight * jnp.sum(state_entropy)
    
    # Expected free energy (to be minimized)
    expected_free_energy = -pragmatic_value - epistemic_value
    
    components = {
        "pragmatic_value": pragmatic_value,
        "epistemic_value": epistemic_value,
        "goal_error": jnp.sqrt(jnp.mean(jnp.square(goal_error)))
    }
    
    return expected_free_energy, components


@partial(jit, static_argnums=(3,))
def compute_gaussian_entropy(
    mean: jnp.ndarray,
    log_variance: jnp.ndarray,
    epsilon: float = 1e-8,
    return_total: bool = True
) -> jnp.ndarray:
    """
    Compute entropy of Gaussian distribution.
    
    H = 0.5 * ln(2*pi*e*variance)
    
    Args:
        mean: Mean of distribution
        log_variance: Log variance of distribution
        epsilon: Small constant for numerical stability
        return_total: If True, return sum; if False, return per-dimension
        
    Returns:
        Entropy value(s)
    """
    entropy = 0.5 * (jnp.log(2.0 * jnp.pi * jnp.e) + log_variance)
    
    if return_total:
        return jnp.sum(entropy)
    return entropy


@jit
def compute_kl_divergence(
    mean_q: jnp.ndarray,
    log_var_q: jnp.ndarray,
    mean_p: jnp.ndarray,
    log_var_p: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute KL divergence between two Gaussian distributions.
    
    KL(q||p) where q is posterior, p is prior
    
    Args:
        mean_q: Mean of q (posterior)
        log_var_q: Log variance of q
        mean_p: Mean of p (prior)
        log_var_p: Log variance of p
        
    Returns:
        KL divergence value
    """
    var_q = jnp.exp(log_var_q)
    var_p = jnp.exp(log_var_p)
    
    kl = 0.5 * (
        log_var_p - log_var_q
        + var_q / var_p
        + jnp.square(mean_q - mean_p) / var_p
        - 1.0
    )
    
    return jnp.sum(kl)


@jit
def compute_state_gradients(
    observation: jnp.ndarray,
    state: jnp.ndarray,
    generative_weights: jnp.ndarray,
    prior_mean: jnp.ndarray,
    observation_precision: float = 1.0,
    state_precision: float = 1.0
) -> jnp.ndarray:
    """
    Compute ∂F/∂μ for gradient descent on beliefs.

    ∂F/∂μ = σ_o^(-2) * W_gen^T * (g(μ) - o) + σ_s^(-2) * (μ - μ_0)

    Args:
        observation: Observed data (sensory input)
        state: Current state beliefs
        generative_weights: Generative model weights W_gen
        prior_mean: Prior belief mean μ_0
        observation_precision: Observation precision σ_o^(-2)
        state_precision: State precision σ_s^(-2)

    Returns:
        Gradient of free energy w.r.t. state
    """
    prediction = jnp.dot(state, generative_weights)
    pred_error = observation - prediction

    # Gradient of accuracy term
    grad_accuracy = -observation_precision * jnp.dot(pred_error, generative_weights.T)

    # Gradient of complexity term
    grad_complexity = state_precision * (state - prior_mean)

    return grad_accuracy + grad_complexity


@jit
def compute_state_entropy(
    mean: jnp.ndarray,
    log_variance: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute entropy H[q(μ)] of Gaussian state distribution.

    H = 0.5 * d * log(2πe) + 0.5 * Σ log(σ²)

    Args:
        mean: Mean of distribution
        log_variance: Log variance of distribution

    Returns:
        Entropy value (scalar)
    """
    n_dims = mean.shape[-1]
    const_term = 0.5 * n_dims * jnp.log(2.0 * jnp.pi * jnp.e)
    var_term = 0.5 * jnp.sum(log_variance)
    return const_term + var_term


@jit
def compute_information_gain(
    prior_mean: jnp.ndarray,
    prior_log_var: jnp.ndarray,
    posterior_mean: jnp.ndarray,
    posterior_log_var: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute information gain I[s;o] = H[p(s)] - H[p(s|o)].

    Measures reduction in uncertainty about states after observation.

    Args:
        prior_mean: Prior state mean
        prior_log_var: Prior state log variance
        posterior_mean: Posterior state mean
        posterior_log_var: Posterior state log variance

    Returns:
        Information gain (non-negative scalar)
    """
    prior_entropy = compute_state_entropy(prior_mean, prior_log_var)
    posterior_entropy = compute_state_entropy(posterior_mean, posterior_log_var)
    return prior_entropy - posterior_entropy


@jit
def compute_policy_posterior(
    expected_free_energies: jnp.ndarray,
    temperature: float = 1.0
) -> jnp.ndarray:
    """
    Compute policy posterior q(π) = softmax(-G(π)/γ).

    Args:
        expected_free_energies: G(π) for each policy
        temperature: γ controls exploration (high) vs exploitation (low)

    Returns:
        Policy probabilities summing to 1
    """
    log_probs = -expected_free_energies / temperature
    log_probs = log_probs - jnp.max(log_probs)  # Numerical stability
    probs = jnp.exp(log_probs)
    return probs / jnp.sum(probs)


@jit
def compute_ambiguity(
    predicted_observation: jnp.ndarray,
    observation_log_variance: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute ambiguity: expected conditional entropy E[H[p(o|s,π)]].

    Measures uncertainty about observations given states.

    Args:
        predicted_observation: Predicted observation mean
        observation_log_variance: Log variance of observation distribution

    Returns:
        Ambiguity term (scalar)
    """
    n_dims = predicted_observation.shape[-1]
    const_term = 0.5 * n_dims * jnp.log(2.0 * jnp.pi * jnp.e)
    var_term = 0.5 * jnp.sum(observation_log_variance)
    return const_term + var_term


