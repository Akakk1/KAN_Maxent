"""KAN-MaxEnt: additive B-spline MaxEnt / IPP for presence-only SDM."""

__version__ = "0.1.0"

from kanmaxent.losses.ipp import ipp_nll, log_partition
from kanmaxent.models.additive_spline_kan import AdditiveSplineKAN

__all__ = [
    "__version__",
    "ipp_nll",
    "log_partition",
    "AdditiveSplineKAN",
]
