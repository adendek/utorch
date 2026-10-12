"""Trainable network parameter abstraction."""

from __future__ import annotations

from typing import Any

from utorch.autograd import Tensor


class Parameter(Tensor):
    """A Tensor designated as a trainable model parameter.

    Parameters are automatically detected and registered by `nn.Module` instances
    when assigned as attributes, making them discoverable via `module.parameters()`.

    Args:
        data: Numerical data (NumPy array, nested list, or scalar).
        name: Optional human-readable identifier for diagramming and debugging.

    Examples:
        ```python
        import numpy as np
        import utorch as ut
        from utorch.nn import Parameter

        # Initialize a trainable weight matrix
        w = Parameter(np.random.randn(3, 2), name="weight")
        print(isinstance(w, ut.Tensor))  # True
        ```
    """

    def __init__(self, data: Any, name: str | None = None) -> None:
        """Initialize a model parameter."""
        super().__init__(value=data, name=name)
        self.is_parameter: bool = True
