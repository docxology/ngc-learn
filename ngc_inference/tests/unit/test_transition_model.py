"""
Unit tests for transition models p(s'|s,a).
"""

import pytest
import jax.numpy as jnp
from jax import random

from ngc_inference.core.transition_model import (
    TransitionModel,
    DiscreteTransitionModel,
    ContinuousTransitionModel
)


@pytest.mark.unit
class TestDiscreteTransitionModel:
    """Test discrete transition model."""

    def setup_method(self):
        """Setup test fixtures."""
        self.key = random.PRNGKey(42)
        self.n_states = 5
        self.n_actions = 3

    def test_initialization(self):
        """Test model initializes correctly."""
        model = DiscreteTransitionModel(self.n_states, self.n_actions)
        assert model.n_states == self.n_states
        assert model.n_actions == self.n_actions

    def test_transition_matrices_normalized(self):
        """Test T[a] rows sum to 1."""
        model = DiscreteTransitionModel(self.n_states, self.n_actions)
        for a in range(self.n_actions):
            T = model.get_transition_matrix(a)
            row_sums = jnp.sum(T, axis=1)
            assert jnp.allclose(row_sums, 1.0)

    def test_predict_next_state(self):
        """Test next state prediction."""
        model = DiscreteTransitionModel(self.n_states, self.n_actions)
        state = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])  # One-hot
        action = 1

        next_state_mean, next_state_var = model.predict_next_state(state, action)
        assert next_state_mean.shape == state.shape
        assert jnp.all(next_state_var >= 0)

    def test_learning_from_experience(self):
        """Test transition matrix updates from data."""
        model = DiscreteTransitionModel(self.n_states, self.n_actions, learning_rate=0.1)

        # Generate deterministic transition: state 0, action 0 -> state 1
        for _ in range(50):
            state = jnp.zeros(self.n_states)
            state = state.at[0].set(1.0)
            next_state = jnp.zeros(self.n_states)
            next_state = next_state.at[1].set(1.0)
            model.update(state, 0, next_state)

        T = model.get_transition_matrix(0)
        # T[0,1] should be high
        assert T[0, 1] > 0.5


@pytest.mark.unit
class TestContinuousTransitionModel:
    """Test continuous transition model."""

    def setup_method(self):
        """Setup test fixtures."""
        self.key = random.PRNGKey(42)
        self.n_states = 4
        self.n_actions = 2

    def test_initialization(self):
        """Test continuous model initializes."""
        model = ContinuousTransitionModel(
            self.n_states,
            self.n_actions,
            hidden_dims=[8]
        )
        assert model.n_states == self.n_states
        assert model.n_actions == self.n_actions

    def test_predict_shape(self):
        """Test prediction shape."""
        model = ContinuousTransitionModel(self.n_states, self.n_actions)
        state = random.normal(self.key, (self.n_states,))
        action = random.normal(self.key, (self.n_actions,))

        next_mean, next_var = model.predict_next_state(state, action)
        assert next_mean.shape == state.shape
        assert next_var.shape == state.shape

    def test_variance_positive(self):
        """Test predicted variance is positive."""
        model = ContinuousTransitionModel(self.n_states, self.n_actions)
        state = random.normal(self.key, (self.n_states,))
        action = random.normal(self.key, (self.n_actions,))

        _, next_var = model.predict_next_state(state, action)
        assert jnp.all(next_var > 0)

    def test_learning_improves_prediction(self):
        """Test learning reduces prediction error."""
        model = ContinuousTransitionModel(
            self.n_states,
            self.n_actions,
            learning_rate=0.01
        )

        # Generate data: next_state = state + action
        initial_errors = []
        final_errors = []

        for _ in range(100):
            state = random.normal(self.key, (self.n_states,))
            action = random.normal(self.key, (self.n_actions,))
            next_state = state + jnp.concatenate([action, jnp.zeros(self.n_states - self.n_actions)])

            pred_before, _ = model.predict_next_state(state, action)
            initial_errors.append(jnp.mean((pred_before - next_state)**2))

            model.update(state, action, next_state)

            pred_after, _ = model.predict_next_state(state, action)
            final_errors.append(jnp.mean((pred_after - next_state)**2))

        # Error should decrease
        assert jnp.mean(jnp.array(final_errors[-10:])) < jnp.mean(jnp.array(initial_errors[:10]))
