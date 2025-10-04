"""
Active Inference agent with full EFE computation and policy selection.

Implements complete Active Inference with:
- Variational free energy minimization via gradient descent
- Expected free energy computation with pragmatic + epistemic values
- Policy posterior sampling q(π) = softmax(-G/γ)
- Learnable transition models p(s'|s,a)
- Support for discrete and continuous action spaces
"""

from typing import Dict, Tuple, Optional, List, Union
import jax.numpy as jnp
from jax import random, jit
from ngclearn import Context
from ngclearn.components import RateCell, GaussianErrorCell, DenseSynapse
from ngcsimlib.compilers.process import Process

from ngc_inference.core.free_energy import (
    compute_free_energy,
    compute_state_gradients,
    compute_expected_free_energy,
    compute_policy_posterior,
    compute_state_entropy,
    compute_information_gain
)
from ngc_inference.core.transition_model import TransitionModel, DiscreteTransitionModel, ContinuousTransitionModel
from ngc_inference.utils.logging_config import get_logger

logger = get_logger(__name__)


class ActiveInferenceAgent:
    """
    Full Active Inference agent with VFE + EFE.

    Implements complete Active Inference loop:
    1. Perceive: Minimize VFE via gradient descent on beliefs
    2. Plan: Compute EFE for all policies
    3. Act: Sample from policy posterior q(π) = softmax(-G/γ)
    4. Learn: Update transition and generative models from experience
    """

    def __init__(
        self,
        n_states: int,
        n_observations: int,
        n_actions: int,
        action_space: str = "discrete",
        learning_rate_states: float = 0.1,
        learning_rate_params: float = 0.01,
        observation_precision: float = 1.0,
        state_precision: float = 1.0,
        policy_temperature: float = 1.0,
        transition_model_type: str = "discrete",
        transition_hidden_dims: Optional[List[int]] = None,
        seed: int = 42
    ):
        """
        Initialize Active Inference agent.

        Args:
            n_states: State space dimensionality
            n_observations: Observation space dimensionality
            n_actions: Action space dimensionality (count for discrete, dim for continuous)
            action_space: "discrete" or "continuous"
            learning_rate_states: Learning rate for belief updates (η_μ)
            learning_rate_params: Learning rate for parameter updates (η_θ)
            observation_precision: Observation precision σ_o^(-2)
            state_precision: State precision σ_s^(-2)
            policy_temperature: Temperature γ for policy posterior
            transition_model_type: "discrete" or "continuous"
            transition_hidden_dims: Hidden layer sizes for continuous transition model
            seed: Random seed
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.action_space = action_space
        self.learning_rate_states = learning_rate_states
        self.learning_rate_params = learning_rate_params
        self.observation_precision = observation_precision
        self.state_precision = state_precision
        self.policy_temperature = policy_temperature
        self.key = random.PRNGKey(seed)

        # Initialize transition model
        if transition_model_type == "discrete":
            self.transition_model = DiscreteTransitionModel(
                n_states, n_actions, learning_rate=learning_rate_params, seed=seed
            )
        elif transition_model_type == "continuous":
            self.transition_model = ContinuousTransitionModel(
                n_states, n_actions,
                hidden_dims=transition_hidden_dims,
                learning_rate=learning_rate_params,
                seed=seed
            )
        else:
            raise ValueError(f"Unknown transition model type: {transition_model_type}")

        # Initialize generative model weights
        self.W_gen = random.normal(self.key, (n_states, n_observations)) * 0.1

        # Initialize prior beliefs
        self.prior_mean = jnp.zeros(n_states)
        self.prior_log_var = jnp.zeros(n_states)

        # Current beliefs (posterior)
        self.current_beliefs = jnp.zeros(n_states)
        self.current_beliefs_log_var = jnp.zeros(n_states)

        logger.info(f"Initialized ActiveInferenceAgent")
        logger.info(f"  States: {n_states}, Observations: {n_observations}, Actions: {n_actions}")
        logger.info(f"  Action space: {action_space}, Transition: {transition_model_type}")

    def infer(self, observation: jnp.ndarray, n_steps: int = 30) -> Tuple[jnp.ndarray, Dict]:
        """
        Perform inference: minimize VFE via gradient descent on beliefs.

        Args:
            observation: Observed data (batch_size, n_observations)
            n_steps: Number of inference iterations

        Returns:
            final_beliefs: Inferred state beliefs
            metrics: Dictionary with free energy trajectory and components
        """
        beliefs = self.current_beliefs.copy()
        beliefs_log_var = self.current_beliefs_log_var.copy()

        fe_trajectory = []

        for step in range(n_steps):
            # Compute free energy and gradients
            prediction = jnp.dot(beliefs, self.W_gen)
            prior_mean = self.prior_mean
            prior_log_var = self.prior_log_var

            fe, fe_components = compute_free_energy(
                observation, prediction, prior_mean, beliefs,
                self.observation_precision, self.state_precision
            )

            # Compute gradients for belief updates
            grad = compute_state_gradients(
                observation, beliefs, self.W_gen, prior_mean,
                self.observation_precision, self.state_precision
            )

            # Gradient descent update
            beliefs = beliefs - self.learning_rate_states * grad

            # Update beliefs variance (simplified)
            beliefs_log_var = jnp.log(self.state_precision)  # Constant for now

            fe_trajectory.append(float(fe))

        # Update current beliefs
        self.current_beliefs = beliefs
        self.current_beliefs_log_var = beliefs_log_var

        # Compute final metrics
        final_prediction = jnp.dot(beliefs, self.W_gen)
        final_fe, final_components = compute_free_energy(
            observation, final_prediction, self.prior_mean, beliefs,
            self.observation_precision, self.state_precision
        )

        metrics = {
            "final_beliefs": beliefs,
            "final_prediction": final_prediction,
            "free_energy": final_fe,
            "free_energy_trajectory": fe_trajectory,
            "fe_components": final_components,
            "beliefs_variance": jnp.exp(beliefs_log_var)
        }

        return beliefs, metrics

    def evaluate_policy(
        self,
        state: jnp.ndarray,
        action: Union[int, jnp.ndarray],
        preferred_observation: jnp.ndarray
    ) -> Tuple[float, Dict]:
        """
        Compute expected free energy G(π) for a policy.

        G = -Pragmatic - Epistemic

        Args:
            state: Current state beliefs
            action: Action to evaluate
            preferred_observation: Goal observation o*

        Returns:
            expected_free_energy: G(π) value
            components: Dictionary with pragmatic and epistemic terms
        """
        # Predict next state distribution
        if self.action_space == "discrete":
            next_state_mean, next_state_var = self.transition_model.predict_next_state(state, action)
        else:
            next_state_mean, next_state_var = self.transition_model.predict_next_state(state, action)

        # Predict next observation
        next_obs = jnp.dot(next_state_mean, self.W_gen)

        # Pragmatic value: -E[log p(o*)] = -0.5 * σ_o^(-2) * ||next_obs - o*||²
        goal_error = next_obs - preferred_observation
        pragmatic_value = -0.5 * self.observation_precision * jnp.sum(goal_error ** 2)

        # Epistemic value: Expected information gain I[s';o'] = H[p(s')] - H[p(s'|o')]
        # For simplicity, use state entropy as proxy for epistemic value
        prior_entropy = compute_state_entropy(next_state_mean, jnp.log(next_state_var))
        epistemic_value = -prior_entropy  # Information-seeking drive

        # Expected free energy (to be minimized)
        expected_free_energy = -pragmatic_value - epistemic_value

        components = {
            "pragmatic_value": pragmatic_value,
            "epistemic_value": epistemic_value,
            "next_state_mean": next_state_mean,
            "next_state_var": next_state_var,
            "predicted_observation": next_obs,
            "goal_error": jnp.sqrt(jnp.mean(goal_error ** 2))
        }

        return expected_free_energy, components

    def select_action(
        self,
        observation: jnp.ndarray,
        preferred_observation: jnp.ndarray,
        sample: bool = True
    ) -> Tuple[Union[int, jnp.ndarray], Dict]:
        """
        Select action via policy posterior q(π) = softmax(-G(π)/γ).

        Args:
            observation: Current observation
            preferred_observation: Goal observation
            sample: Whether to sample from posterior (True) or take argmax (False)

        Returns:
            selected_action: Chosen action
            metrics: Dictionary with EFEs, policy posterior, etc.
        """
        # Infer current state
        beliefs, _ = self.infer(observation)

        # Evaluate all possible actions
        efes = []
        policy_components = []

        if self.action_space == "discrete":
            actions = list(range(self.n_actions))
        else:
            # For continuous, sample a few actions (simplified)
            self.key, subkey = random.split(self.key)
            actions = [random.normal(subkey, (self.n_actions,)) for _ in range(5)]

        for action in actions:
            efe, components = self.evaluate_policy(beliefs, action, preferred_observation)
            efes.append(efe)
            policy_components.append(components)

        # Compute policy posterior
        efes_array = jnp.array(efes)
        policy_posterior = compute_policy_posterior(efes_array, self.policy_temperature)

        # Select action
        if sample:
            # Sample from posterior
            self.key, subkey = random.split(self.key)
            action_idx = random.categorical(subkey, jnp.log(policy_posterior))
            selected_action = actions[action_idx]
            best_idx = jnp.argmin(efes_array)  # For logging
        else:
            # Take action with minimum EFE
            best_idx = jnp.argmin(efes_array)
            selected_action = actions[best_idx]

        metrics = {
            "selected_action": selected_action,
            "expected_free_energies": efes_array,
            "policy_posterior": policy_posterior,
            "beliefs": beliefs,
            "efe_components": policy_components
        }

        logger.debug(f"Action selected: EFE={efes_array[best_idx]:.4f}")

        return selected_action, metrics

    def learn_generative_model(self, observations: jnp.ndarray, n_epochs: int = 100) -> Dict:
        """
        Learn generative model p(o|s) from observations.

        Args:
            observations: Training data (n_samples, n_observations)
            n_epochs: Number of training epochs

        Returns:
            Training metrics
        """
        losses = []

        for epoch in range(n_epochs):
            epoch_loss = 0.0

            for obs in observations:
                # Inference to get beliefs
                beliefs, _ = self.infer(obs)

                # Compute prediction error
                prediction = jnp.dot(beliefs, self.W_gen)
                error = obs - prediction

                # Hebbian update: ΔW = η * μ^T * ε
                self.W_gen += self.learning_rate_params * jnp.outer(beliefs, error)

                # Accumulate loss
                epoch_loss += jnp.mean(error ** 2)

            epoch_loss /= len(observations)
            losses.append(epoch_loss)

            if (epoch + 1) % max(1, n_epochs // 10) == 0:
                logger.info(f"Generative model epoch {epoch+1}/{n_epochs}: Loss={epoch_loss:.4f}")

        return {
            "losses": losses,
            "final_loss": losses[-1],
            "final_weights": self.W_gen.copy()
        }

    def learn_transition_model(self, trajectories: List[Tuple[jnp.ndarray, Union[int, jnp.ndarray], jnp.ndarray]]) -> Dict:
        """
        Learn transition model p(s'|s,a) from experience.

        Args:
            trajectories: List of (state, action, next_state) tuples

        Returns:
            Learning metrics
        """
        initial_errors = []
        final_errors = []

        for i, (state, action, next_state) in enumerate(trajectories):
            # Compute prediction before update
            pred_mean, _ = self.transition_model.predict_next_state(state, action)
            initial_errors.append(jnp.mean((pred_mean - next_state) ** 2))

            # Update model
            self.transition_model.update(state, action, next_state)

            # Compute prediction after update
            pred_mean_after, _ = self.transition_model.predict_next_state(state, action)
            final_errors.append(jnp.mean((pred_mean_after - next_state) ** 2))

        return {
            "initial_errors": initial_errors,
            "final_errors": final_errors,
            "improvement": jnp.mean(jnp.array(initial_errors)) - jnp.mean(jnp.array(final_errors))
        }

    def reset(self):
        """Reset agent state."""
        self.current_beliefs = jnp.zeros(self.n_states)
        self.current_beliefs_log_var = jnp.zeros(self.n_states)

    def get_beliefs(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get current beliefs and their variance."""
        return self.current_beliefs, jnp.exp(self.current_beliefs_log_var)
