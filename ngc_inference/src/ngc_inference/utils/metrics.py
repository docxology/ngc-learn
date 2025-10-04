"""
Metrics computation utilities for Active Inference simulations.

Provides metric tracking and computation functions for inference and learning.
"""

from typing import Dict, List, Optional
import jax.numpy as jnp
from jax import jit


class InferenceMetrics:
    """
    Container for tracking inference metrics over time.
    
    Tracks free energy, prediction errors, complexity, and accuracy.
    """
    
    def __init__(self):
        """Initialize metrics container."""
        self.free_energy: List[float] = []
        self.prediction_error: List[float] = []
        self.complexity: List[float] = []
        self.accuracy: List[float] = []
        
    def add(
        self,
        free_energy: Optional[float] = None,
        prediction_error: Optional[float] = None,
        complexity: Optional[float] = None,
        accuracy: Optional[float] = None,
        fe: Optional[float] = None,
        pe: Optional[float] = None,
        **kwargs
    ):
        """
        Add metrics for current step.
        
        Args:
            free_energy: Free energy value (or use fe)
            prediction_error: Prediction error magnitude (or use pe)
            complexity: Complexity term
            accuracy: Accuracy term
            fe: Alias for free_energy
            pe: Alias for prediction_error
            **kwargs: Additional metric values to allow flexible usage
        """
        # Support both full names and abbreviations
        fe_val = free_energy if free_energy is not None else fe
        pe_val = prediction_error if prediction_error is not None else pe
        
        if fe_val is not None:
            self.free_energy.append(fe_val)
        if pe_val is not None:
            self.prediction_error.append(pe_val)
        if complexity is not None:
            self.complexity.append(complexity)
        if accuracy is not None:
            self.accuracy.append(accuracy)
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary of summary metrics
        """
        if not self.free_energy:
            return {}
        
        return {
            "mean_free_energy": float(jnp.mean(jnp.array(self.free_energy))),
            "final_free_energy": self.free_energy[-1],
            "mean_prediction_error": float(jnp.mean(jnp.array(self.prediction_error))),
            "final_prediction_error": self.prediction_error[-1],
        }
    
    def reset(self):
        """Reset all tracked metrics."""
        self.free_energy = []
        self.prediction_error = []
        self.complexity = []
        self.accuracy = []


@jit
def compute_rmse(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """
    Compute root mean squared error.
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        RMSE value
    """
    return jnp.sqrt(jnp.mean(jnp.square(predictions - targets)))


@jit
def compute_mae(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """
    Compute mean absolute error.
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        MAE value
    """
    return jnp.mean(jnp.abs(predictions - targets))


@jit
def compute_r2_score(predictions: jnp.ndarray, targets: jnp.ndarray) -> float:
    """
    Compute R² (coefficient of determination).
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        R² score
    """
    ss_res = jnp.sum(jnp.square(targets - predictions))
    ss_tot = jnp.sum(jnp.square(targets - jnp.mean(targets)))
    return 1.0 - (ss_res / (ss_tot + 1e-8))


def compute_metrics(
    observations: jnp.ndarray,
    predictions: jnp.ndarray,
    beliefs: jnp.ndarray,
    priors: jnp.ndarray
) -> Dict[str, float]:
    """
    Compute comprehensive inference metrics.
    
    Args:
        observations: Observed data
        predictions: Model predictions
        beliefs: Inferred beliefs
        priors: Prior beliefs
        
    Returns:
        Dictionary of computed metrics
    """
    return {
        "rmse": float(compute_rmse(predictions, observations)),
        "mae": float(compute_mae(predictions, observations)),
        "r2_score": float(compute_r2_score(predictions, observations)),
        "prediction_error": float(compute_rmse(predictions, observations)),
        "state_divergence": float(jnp.mean(jnp.abs(beliefs - priors))),
        "belief_magnitude": float(jnp.mean(jnp.abs(beliefs))),
        "prior_divergence": float(jnp.mean(jnp.abs(beliefs - priors))),
        "correlation": float(jnp.corrcoef(predictions.flatten(), observations.flatten())[0, 1]),
    }



