"""
NGC Inference: Active Inference framework using ngclearn.

This package provides tools for building Active Inference agents using variational
free energy minimization with ngclearn's neurobiologically plausible components.
"""

__version__ = "0.1.0"

from ngc_inference.core.free_energy import (
    compute_free_energy,
    compute_expected_free_energy,
    compute_prediction_error,
)
from ngc_inference.core.inference import (
    VariationalInferenceAgent,
    ActiveInferenceAgent,
)

__all__ = [
    "compute_free_energy",
    "compute_expected_free_energy",
    "compute_prediction_error",
    "VariationalInferenceAgent",
    "ActiveInferenceAgent",
]

