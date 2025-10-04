"""
Unit tests for metrics computation.
"""

import pytest
import jax.numpy as jnp
from jax import random

from ngc_inference.utils.metrics import (
    InferenceMetrics,
    compute_metrics,
    compute_rmse,
    compute_mae,
    compute_r2_score
)


@pytest.mark.unit
class TestMetrics:
    """Test metrics computation utilities."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.key = random.PRNGKey(42)
        
    def test_inference_metrics_container(self):
        """Test InferenceMetrics container."""
        metrics = InferenceMetrics()
        
        # Add some data
        metrics.add(fe=1.0, pe=0.5, complexity=0.3, accuracy=0.7)
        metrics.add(fe=0.8, pe=0.4, complexity=0.2, accuracy=0.6)
        
        # Get summary
        summary = metrics.get_summary()
        
        assert "mean_free_energy" in summary
        assert "final_free_energy" in summary
        assert summary["final_free_energy"] == 0.8
        
        # Reset
        metrics.reset()
        assert len(metrics.free_energy) == 0
        
    def test_compute_metrics(self):
        """Test comprehensive metrics computation."""
        observations = random.normal(self.key, (10, 5))
        predictions = observations + 0.1 * random.normal(self.key, (10, 5))
        beliefs = random.normal(self.key, (10, 3))
        priors = jnp.zeros((10, 3))
        
        metrics = compute_metrics(observations, predictions, beliefs, priors)
        
        assert "prediction_error" in metrics
        assert "state_divergence" in metrics
        assert "correlation" in metrics
        
        # All metrics should be finite
        for value in metrics.values():
            assert jnp.isfinite(value)
        
    def test_rmse(self):
        """Test RMSE computation."""
        predictions = jnp.array([[1.0, 2.0, 3.0]])
        targets = jnp.array([[1.1, 2.1, 3.1]])
        
        rmse = compute_rmse(predictions, targets)
        
        expected_rmse = jnp.sqrt(jnp.mean(jnp.square(0.1)))
        assert jnp.allclose(rmse, expected_rmse)
        
    def test_mae(self):
        """Test MAE computation."""
        predictions = jnp.array([[1.0, 2.0, 3.0]])
        targets = jnp.array([[1.1, 2.2, 3.3]])
        
        mae = compute_mae(predictions, targets)
        
        expected_mae = jnp.mean(jnp.array([0.1, 0.2, 0.3]))
        assert jnp.allclose(mae, expected_mae)
        
    def test_r2_score_perfect(self):
        """Test R² score with perfect predictions."""
        targets = random.normal(self.key, (10, 5))
        predictions = targets
        
        r2 = compute_r2_score(predictions, targets)
        
        assert jnp.allclose(r2, 1.0, atol=1e-5)
        
    def test_r2_score_mean_prediction(self):
        """Test R² score with mean predictions."""
        targets = random.normal(self.key, (10, 5))
        predictions = jnp.ones_like(targets) * jnp.mean(targets)
        
        r2 = compute_r2_score(predictions, targets)
        
        # R² should be close to 0 for mean predictions
        assert jnp.allclose(r2, 0.0, atol=0.1)




