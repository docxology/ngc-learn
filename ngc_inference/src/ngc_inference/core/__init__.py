"""
Core Active Inference computation modules.
"""

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

