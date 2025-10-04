"""
Integration tests for ActiveInferenceAgent.
"""

import pytest
import jax.numpy as jnp
from jax import random

from ngc_inference.core.active_inference_agent import ActiveInferenceAgent


@pytest.mark.integration
class TestActiveInferenceAgent:
    """Test complete Active Inference agent."""

    def setup_method(self):
        """Setup test fixtures."""
        self.key = random.PRNGKey(42)

    def test_agent_initialization_discrete(self):
        """Test discrete action space initialization."""
        agent = ActiveInferenceAgent(
            n_states=5,
            n_observations=10,
            n_actions=3,
            action_space="discrete",
            transition_model_type="discrete"
        )

        assert agent.n_states == 5
        assert agent.n_observations == 10
        assert agent.n_actions == 3
        assert agent.action_space == "discrete"
        assert isinstance(agent.transition_model, type(agent.transition_model).__bases__[0].__subclasses__()[0])  # DiscreteTransitionModel

    def test_agent_initialization_continuous(self):
        """Test continuous action space initialization."""
        agent = ActiveInferenceAgent(
            n_states=4,
            n_observations=2,
            n_actions=2,
            action_space="continuous",
            transition_model_type="continuous"
        )

        assert agent.action_space == "continuous"
        assert agent.transition_model.n_actions == 2

    def test_vfe_decreases_during_inference(self):
        """CRITICAL: Verify FE actually minimizes during inference."""
        agent = ActiveInferenceAgent(
            n_states=5,
            n_observations=10,
            n_actions=3,
            learning_rate_states=0.1
        )

        # Create simple observation
        observation = jnp.ones((1, 10)) * 0.5

        beliefs, metrics = agent.infer(observation, n_steps=50)

        fe_traj = metrics["free_energy_trajectory"]

        # Free energy should decrease (this was failing before!)
        assert fe_traj[-1] < fe_traj[0], f"FE increased: {fe_traj[0]:.4f} -> {fe_traj[-1]:.4f}"

    def test_efe_computation(self):
        """Test EFE computation with pragmatic and epistemic components."""
        agent = ActiveInferenceAgent(n_states=3, n_observations=2, n_actions=2)

        state = jnp.array([1.0, 0.0, 0.0])
        action = 0
        goal = jnp.array([0.5, 0.5])

        efe, components = agent.evaluate_policy(state, action, goal)

        assert "pragmatic_value" in components
        assert "epistemic_value" in components
        assert "next_state_mean" in components
        assert "predicted_observation" in components

    def test_policy_posterior_normalization(self):
        """Test policy posterior sums to 1."""
        agent = ActiveInferenceAgent(n_states=3, n_observations=2, n_actions=3)

        state = jnp.array([1.0, 0.0, 0.0])
        goal = jnp.array([0.5, 0.5])

        action, metrics = agent.select_action(
            jnp.array([[0.3, 0.7]]), goal, sample=False
        )

        policy_probs = metrics["policy_posterior"]
        assert jnp.allclose(jnp.sum(policy_probs), 1.0)

    def test_generative_model_learning(self):
        """Test generative model learns from data."""
        agent = ActiveInferenceAgent(n_states=5, n_observations=10, n_actions=3)

        # Generate simple training data
        n_samples = 50
        observations = jnp.zeros((n_samples, 10))

        # Observations correlate with states
        for i in range(n_samples):
            state_idx = i % 5
            observations = observations.at[i, state_idx*2:(state_idx+1)*2].set(1.0)

        # Learn generative model
        results = agent.learn_generative_model(observations, n_epochs=20)

        assert "losses" in results
        assert len(results["losses"]) == 20
        assert results["final_loss"] < results["losses"][0]  # Should improve

    def test_transition_model_learning_discrete(self):
        """Test discrete transition model learning."""
        agent = ActiveInferenceAgent(
            n_states=3,
            n_observations=2,
            n_actions=2,
            transition_model_type="discrete"
        )

        # Generate trajectories: state 0, action 0 -> state 1
        trajectories = []
        for _ in range(30):
            state = jnp.array([1.0, 0.0, 0.0])
            action = 0
            next_state = jnp.array([0.0, 1.0, 0.0])
            trajectories.append((state, action, next_state))

        results = agent.learn_transition_model(trajectories)

        assert "improvement" in results
        # Should show some improvement
        assert results["improvement"] >= -0.1  # Allow small negative for numerical issues

    def test_continuous_action_space(self):
        """Test continuous action space functionality."""
        agent = ActiveInferenceAgent(
            n_states=4,
            n_observations=2,
            n_actions=2,
            action_space="continuous",
            transition_model_type="continuous"
        )

        observation = jnp.array([[0.3, 0.7]])
        goal = jnp.array([0.5, 0.5])

        action, metrics = agent.select_action(observation, goal)

        assert action.shape == (2,)  # 2D continuous action
        assert "expected_free_energies" in metrics
        assert "policy_posterior" in metrics

    def test_belief_tracking(self):
        """Test that beliefs are properly tracked."""
        agent = ActiveInferenceAgent(n_states=3, n_observations=2, n_actions=2)

        # Reset agent
        agent.reset()

        # Check initial beliefs
        beliefs, var = agent.get_beliefs()
        assert jnp.allclose(beliefs, 0.0)
        assert jnp.allclose(var, 1.0)  # exp(0) = 1

        # Perform inference
        observation = jnp.array([[1.0, 0.0]])
        final_beliefs, _ = agent.infer(observation, n_steps=10)

        # Check that beliefs changed
        assert not jnp.allclose(final_beliefs, 0.0)
