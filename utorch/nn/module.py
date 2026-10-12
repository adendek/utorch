"""Base Module class for neural network layers and architectures."""

from __future__ import annotations

from typing import Any

import numpy as np

from utorch.nn.parameter import Parameter


class Module:
    """Base class for all neural network modules and layers.

    Custom models should subclass `Module` and implement `forward()`.
    Modules automatically register child `Module` instances and `Parameter`
    attributes, enabling recursive traversal via `parameters()`.

    Examples:
        ```python
        import utorch as ut
        import utorch.nn as nn

        class TwoLayerMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(784, 128)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                return self.fc2(self.relu(self.fc1(x)))

        model = TwoLayerMLP()
        print(len(model.parameters()))  # 4 (fc1.weight, fc1.bias, fc2.weight, fc2.bias)
        ```
    """

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute forward pass computation. Subclasses must override this."""
        raise NotImplementedError("Subclasses of Module must implement forward().")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call self.forward(*args, **kwargs), mimicking PyTorch's execution syntax."""
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Parameter]:
        """Return an ordered list of all trainable Parameters in this module.

        Recursively discovers parameters stored as direct attributes, inside
        lists/tuples, or within child `Module` submodules.
        """
        params: list[Parameter] = []

        for attr_name in dir(self):
            if attr_name.startswith("__"):
                continue
            val = getattr(self, attr_name)

            if isinstance(val, Parameter):
                params.append(val)
            elif isinstance(val, Module):
                params.extend(val.parameters())
            elif isinstance(val, list | tuple):
                for item in val:
                    if isinstance(item, Parameter):
                        params.append(item)
                    elif isinstance(item, Module):
                        params.extend(item.parameters())

        # Deduplicate preserving order
        seen: set[int] = set()
        unique_params: list[Parameter] = []
        for p in params:
            if id(p) not in seen:
                seen.add(id(p))
                unique_params.append(p)

        return unique_params

    def zero_grad(self) -> None:
        """Reset gradients of all trainable parameters in this module to None."""
        for p in self.parameters():
            p.zero_grad()

    def num_parameters(self) -> int:
        """Return the total number of scalar parameters in the module."""
        return sum(int(np.prod(p.shape)) for p in self.parameters())
