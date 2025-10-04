"""
Integration tests for SimplePredictionAgent.
"""

import pytest
import jax.numpy as jnp
from jax import random

from ngc_inference.simulations.simple_prediction import SimplePredictionAgent


@pytest.mark.integration
class TestSimplePredictionAgent:
    """Integration tests for simple prediction agent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.key = random.PRNGKey(42)
        self.n_observations = 10
        self.n_hidden = 5
        
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = SimplePredictionAgent(
            n_observations=self.n_observations,
            n_hidden=self.n_hidden,
            seed=42
        )
        
        assert agent.n_observations == self.n_observations
        assert agent.n_hidden == self.n_hidden
        assert agent.context is not None
        
    def test_inference_execution(self):
        """Test inference runs without errors."""
        agent = SimplePredictionAgent(
            n_observations=self.n_observations,
            n_hidden=self.n_hidden,
            seed=42
        )
        
        observation = random.normal(self.key, (1, self.n_observations))
        
        beliefs, metrics = agent.infer(observation, n_steps=10)
        
        assert beliefs.shape == (1, self.n_hidden)
        assert "free_energy" in metrics
        assert "free_energy_trajectory" in metrics
        assert len(metrics["free_energy_trajectory"]) == 10
        
    def test_free_energy_decreases(self):
        """Test that free energy decreases during inference."""
        agent = SimplePredictionAgent(
            n_observations=self.n_observations,
            n_hidden=self.n_hidden,
            learning_rate=0.01,
            seed=42
        )
        
        observation = random.normal(self.key, (1, self.n_observations))
        
        beliefs, metrics = agent.infer(observation, n_steps=50)
        
        fe_traj = metrics["free_energy_trajectory"]
        
        # Free energy should generally decrease
        assert fe_traj[-1] < fe_traj[0]
        
    def test_learning_execution(self):
        """Test learning runs without errors."""
        agent = SimplePredictionAgent(
            n_observations=self.n_observations,
            n_hidden=self.n_hidden,
            learning_rate=0.01,
            seed=42
        )
        
        # Generate some training data
        n_samples = 20
        observations = random.normal(self.key, (n_samples, self.n_observations))
        
        training_metrics = agent.learn(
            observations,
            n_epochs=10,
            n_inference_steps=10,
            verbose=False
        )
        
        assert "losses" in training_metrics
        assert len(training_metrics["losses"]) == 10
        
    def test_prediction(self):
        """Test prediction from hidden states."""
        agent = SimplePredictionAgent(
            n_observations=self.n_observations,
            n_hidden=self.n_hidden,
            seed=42
        )
        
        hidden_states = random.normal(self.key, (1, self.n_hidden))
        
        prediction = agent.predict(hidden_states)
        
        assert prediction.shape == (1, self.n_observations)
        
    def test_batch_processing(self):
        """Test agent handles batch processing correctly."""
        batch_size = 4
        agent = SimplePredictionAgent(
            n_observations=self.n_observations,
            n_hidden=self.n_hidden,
            batch_size=batch_size,
            seed=42
        )
        
        observation = random.normal(self.key, (batch_size, self.n_observations))
        
        beliefs, metrics = agent.infer(observation, n_steps=5)
        
        assert beliefs.shape == (batch_size, self.n_hidden)




