"""
Integration tests for HierarchicalInferenceAgent.
"""

import pytest
import jax.numpy as jnp
from jax import random

from ngc_inference.simulations.hierarchical_inference import HierarchicalInferenceAgent


@pytest.mark.integration
class TestHierarchicalInferenceAgent:
    """Integration tests for hierarchical inference agent."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.key = random.PRNGKey(42)
        self.layer_sizes = [10, 8, 6]  # [obs, hidden1, hidden2]
        
    def test_agent_initialization(self):
        """Test hierarchical agent initializes correctly."""
        agent = HierarchicalInferenceAgent(
            layer_sizes=self.layer_sizes,
            seed=42
        )
        
        assert agent.layer_sizes == self.layer_sizes
        assert agent.n_layers == 2
        assert len(agent.states) == 2
        assert len(agent.predictions) == 2
        assert len(agent.errors) == 2
        
    def test_inference_execution(self):
        """Test hierarchical inference runs without errors."""
        agent = HierarchicalInferenceAgent(
            layer_sizes=self.layer_sizes,
            seed=42
        )
        
        observation = random.normal(self.key, (1, self.layer_sizes[0]))
        
        beliefs, metrics = agent.infer(observation, n_steps=20)
        
        # Check beliefs at each level
        assert len(beliefs) == 2  # Two hidden layers
        assert beliefs[0].shape == (1, self.layer_sizes[1])
        assert beliefs[1].shape == (1, self.layer_sizes[2])
        
        assert "free_energy" in metrics
        assert "free_energy_trajectory" in metrics
        
    def test_hierarchical_free_energy_decreases(self):
        """Test that hierarchical free energy decreases."""
        agent = HierarchicalInferenceAgent(
            layer_sizes=self.layer_sizes,
            learning_rate=0.01,
            seed=42
        )
        
        observation = random.normal(self.key, (1, self.layer_sizes[0]))
        
        beliefs, metrics = agent.infer(observation, n_steps=50)
        
        fe_traj = metrics["free_energy_trajectory"]
        
        # Free energy should generally decrease
        assert fe_traj[-1] < fe_traj[0]
        
    def test_hierarchical_learning(self):
        """Test hierarchical learning."""
        agent = HierarchicalInferenceAgent(
            layer_sizes=self.layer_sizes,
            learning_rate=0.005,
            seed=42
        )
        
        n_samples = 15
        observations = random.normal(self.key, (n_samples, self.layer_sizes[0]))
        
        training_metrics = agent.learn(
            observations,
            n_epochs=10,
            n_inference_steps=20,
            verbose=False
        )
        
        assert "losses" in training_metrics
        assert "weights" in training_metrics
        
    def test_generation(self):
        """Test top-down generation."""
        agent = HierarchicalInferenceAgent(
            layer_sizes=self.layer_sizes,
            seed=42
        )
        
        # Generate from top level
        top_state = random.normal(self.key, (1, self.layer_sizes[-1]))
        
        generated = agent.generate(top_state)
        
        assert len(generated) == 2  # Predictions at both levels
        assert generated[0].shape == (1, self.layer_sizes[0])
        
    def test_deeper_hierarchy(self):
        """Test with deeper hierarchy."""
        deep_layers = [10, 15, 12, 8, 5]
        
        agent = HierarchicalInferenceAgent(
            layer_sizes=deep_layers,
            seed=42
        )
        
        observation = random.normal(self.key, (1, deep_layers[0]))
        
        beliefs, metrics = agent.infer(observation, n_steps=30)
        
        assert len(beliefs) == 4  # Four hidden layers
        assert "free_energy" in metrics




