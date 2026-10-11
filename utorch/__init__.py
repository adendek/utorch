"""utorch: A minimalist educational deep learning and autograd library."""

__version__ = "0.1.0"

from utorch import autograd, nn, utils
from utorch.autograd import Tensor

__all__ = [
    "Tensor",
    "__version__",
    "autograd",
    "nn",
    "utils",
]
