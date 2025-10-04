"""
Transition models for Active Inference: p(s'|s,a).

Implements state transition dynamics for both discrete and continuous action spaces.
"""

from typing import Tuple, Optional
from abc import ABC, abstractmethod
import jax.numpy as jnp
from jax import random, jit
from functools import partial


class TransitionModel(ABC):
    """
    Abstract base class for transition models p(s'|s,a).

    Transition models predict next states given current states and actions,
    essential for computing expected free energy in Active Inference.
    """

    def __init__(self, n_states: int, n_actions: int, seed: int = 42):
        """
        Initialize transition model.

        Args:
            n_states: State space dimensionality
            n_actions: Action space dimensionality
            seed: Random seed
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.key = random.PRNGKey(seed)

    @abstractmethod
    def predict_next_state(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict next state distribution.

        Args:
            state: Current state
            action: Action taken

        Returns:
            next_state_mean: Mean of p(s'|s,a)
            next_state_variance: Variance of p(s'|s,a)
        """
        pass

    @abstractmethod
    def compute_likelihood(
        self,
        next_state: jnp.ndarray,
        state: jnp.ndarray,
        action: jnp.ndarray
    ) -> float:
        """
        Compute p(s'|s,a).

        Args:
            next_state: Observed next state
            state: Current state
            action: Action taken

        Returns:
            Log-likelihood value
        """
        pass

    @abstractmethod
    def update(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        next_state: jnp.ndarray
    ) -> None:
        """
        Update model parameters from experience.

        Args:
            state: Current state
            action: Action taken
            next_state: Observed next state
        """
        pass


class DiscreteTransitionModel(TransitionModel):
    """
    Discrete transition model using transition matrices.

    For discrete state and action spaces, maintains T[a][i,j] = p(s'=j|s=i,a).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.1,
        prior_count: float = 1.0,
        seed: int = 42
    ):
        """
        Initialize discrete transition model.

        Args:
            n_states: Number of discrete states
            n_actions: Number of discrete actions
            learning_rate: Update rate for transitions
            prior_count: Dirichlet prior pseudocount
            seed: Random seed
        """
        super().__init__(n_states, n_actions, seed)
        self.learning_rate = learning_rate
        self.prior_count = prior_count

        # Initialize transition matrices T[a][i,j] = p(s'=j|s=i,a)
        # Start with uniform + prior
        self.transition_matrices = []
        for _ in range(n_actions):
            T = jnp.ones((n_states, n_states)) * prior_count
            T = T / jnp.sum(T, axis=1, keepdims=True)  # Normalize rows
            self.transition_matrices.append(T)

        # Count matrices for learning
        self.counts = [
            jnp.ones((n_states, n_states)) * prior_count
            for _ in range(n_actions)
        ]

    def get_transition_matrix(self, action: int) -> jnp.ndarray:
        """Get transition matrix for specific action."""
        return self.transition_matrices[action]

    def predict_next_state(
        self,
        state: jnp.ndarray,
        action: int
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict next state distribution.

        For discrete states (one-hot encoded), returns:
        - Mean: T[a]^T @ state (probability distribution over next states)
        - Variance: Categorical variance
        """
        T = self.transition_matrices[action]
        next_state_dist = jnp.dot(state, T)  # Matrix-vector product

        # Mean is the probability distribution
        next_state_mean = next_state_dist

        # Variance for categorical: p(1-p)
        next_state_var = next_state_dist * (1.0 - next_state_dist)

        return next_state_mean, next_state_var

    def compute_likelihood(
        self,
        next_state: jnp.ndarray,
        state: jnp.ndarray,
        action: int
    ) -> float:
        """
        Compute log p(s'|s,a) for discrete states.

        Assumes one-hot encoded states.
        """
        T = self.transition_matrices[action]
        state_idx = jnp.argmax(state)
        next_state_idx = jnp.argmax(next_state)

        prob = T[state_idx, next_state_idx]
        return jnp.log(prob + 1e-10)  # Avoid log(0)

    def update(
        self,
        state: jnp.ndarray,
        action: int,
        next_state: jnp.ndarray
    ) -> None:
        """
        Update transition matrix from observed transition.

        Uses running average update rule.
        """
        state_idx = jnp.argmax(state)
        next_state_idx = jnp.argmax(next_state)

        # Update counts
        self.counts[action] = self.counts[action].at[state_idx, next_state_idx].add(1.0)

        # Recompute transition probabilities
        T_new = self.counts[action] / jnp.sum(self.counts[action], axis=1, keepdims=True)

        # Smooth update
        T_old = self.transition_matrices[action]
        self.transition_matrices[action] = (
            (1.0 - self.learning_rate) * T_old + self.learning_rate * T_new
        )


class ContinuousTransitionModel(TransitionModel):
    """
    Continuous transition model using neural network dynamics.

    Models s' = f(s, a) + ε where f is a neural network and ε ~ N(0, Σ).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        hidden_dims: Optional[list] = None,
        learning_rate: float = 0.01,
        noise_std: float = 0.1,
        seed: int = 42
    ):
        """
        Initialize continuous transition model.

        Args:
            n_states: State dimensionality
            n_actions: Action dimensionality
            hidden_dims: Hidden layer sizes for neural network
            learning_rate: Learning rate for weight updates
            noise_std: Standard deviation of transition noise
            seed: Random seed
        """
        super().__init__(n_states, n_actions, seed)
        self.learning_rate = learning_rate
        self.noise_std = noise_std

        if hidden_dims is None:
            hidden_dims = [max(8, (n_states + n_actions) // 2)]
        self.hidden_dims = hidden_dims

        # Initialize network weights: [s, a] -> hidden -> s'
        self.key, *subkeys = random.split(self.key, len(hidden_dims) + 2)

        self.weights = []
        self.biases = []

        # Input layer
        input_dim = n_states + n_actions
        W = random.normal(subkeys[0], (input_dim, hidden_dims[0])) * 0.1
        b = jnp.zeros(hidden_dims[0])
        self.weights.append(W)
        self.biases.append(b)

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            W = random.normal(subkeys[i+1], (hidden_dims[i], hidden_dims[i+1])) * 0.1
            b = jnp.zeros(hidden_dims[i+1])
            self.weights.append(W)
            self.biases.append(b)

        # Output layer
        W_out = random.normal(subkeys[-1], (hidden_dims[-1], n_states)) * 0.1
        b_out = jnp.zeros(n_states)
        self.weights.append(W_out)
        self.biases.append(b_out)

    def _forward(self, state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through network."""
        # Concatenate state and action
        x = jnp.concatenate([state.flatten(), action.flatten()])

        # Hidden layers with tanh activation
        for i in range(len(self.weights) - 1):
            x = jnp.dot(x, self.weights[i]) + self.biases[i]
            x = jnp.tanh(x)

        # Output layer (linear)
        x = jnp.dot(x, self.weights[-1]) + self.biases[-1]

        return x

    def predict_next_state(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predict next state using neural network.

        Returns:
            next_state_mean: f(s, a)
            next_state_variance: Fixed noise variance
        """
        next_state_mean = self._forward(state, action)
        next_state_var = jnp.ones_like(next_state_mean) * (self.noise_std ** 2)

        return next_state_mean, next_state_var

    def compute_likelihood(
        self,
        next_state: jnp.ndarray,
        state: jnp.ndarray,
        action: jnp.ndarray
    ) -> float:
        """
        Compute log p(s'|s,a) under Gaussian assumption.
        """
        pred_mean, pred_var = self.predict_next_state(state, action)
        error = next_state - pred_mean

        # Gaussian log-likelihood
        log_lik = -0.5 * jnp.sum(
            jnp.log(2.0 * jnp.pi * pred_var) + (error ** 2) / pred_var
        )

        return log_lik

    def update(
        self,
        state: jnp.ndarray,
        action: jnp.ndarray,
        next_state: jnp.ndarray
    ) -> None:
        """
        Update network weights via gradient descent on prediction error.
        """
        # Predict
        pred_next_state = self._forward(state, action)

        # Compute error
        error = next_state - pred_next_state

        # Gradient descent update (simplified - not using JAX autodiff for now)
        # In practice, would use jax.grad for proper backprop

        # Output layer gradient
        grad_output = -error / (self.noise_std ** 2)

        # Simple update for output weights (last layer)
        # ΔW = -η * grad * activation^T
        hidden_activation = jnp.tanh(
            jnp.dot(jnp.concatenate([state.flatten(), action.flatten()]),
                   self.weights[0]) + self.biases[0]
        )

        self.weights[-1] = self.weights[-1] - self.learning_rate * jnp.outer(
            hidden_activation, grad_output
        )
