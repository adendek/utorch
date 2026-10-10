"""utorch: A minimalist educational deep learning and autograd library."""

__version__ = "0.1.0"

from utorch import autograd, nn, utils
from utorch.dummy_functions import dummy_function

__all__ = [
    "__version__",
    "autograd",
    "dummy_function",
    "nn",
    "utils",
]
