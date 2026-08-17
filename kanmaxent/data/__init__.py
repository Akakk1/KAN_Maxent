"""Data contracts and preprocessing helpers."""

from kanmaxent.data.synthetic import make_linear_scenario, make_nonlinear_scenario, make_uneven_batch_scenario
from kanmaxent.data.ginkgo_io import load_ginkgo, split_outer_fold, env_columns
from kanmaxent.data.preprocess import FoldPreprocessor

__all__ = [
    "make_linear_scenario",
    "make_nonlinear_scenario",
    "make_uneven_batch_scenario",
    "load_ginkgo",
    "split_outer_fold",
    "env_columns",
    "FoldPreprocessor",
]
