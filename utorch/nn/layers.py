"""Neural network layers, activation modules, and sequential containers."""

from __future__ import annotations

from typing import Any

import numpy as np

import utorch as ut
from utorch.nn.module import Module
from utorch.nn.parameter import Parameter


class Linear(Module):
    """Fully connected linear transformation layer: Y = X @ W^T + b.

    Weights are initialized using Kaiming (He) uniform initialization:
        W ~ Uniform(-1 / sqrt(in_features), 1 / sqrt(in_features))

    Args:
        in_features: Number of input features (D_in).
        out_features: Number of output features (D_out).
        bias: If True, adds a learnable bias vector of shape (D_out,).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name: str | None = None,
    ) -> None:
        """Initialize a Linear layer."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.__initialize(bias, in_features, name, out_features)

    def __initialize(
        self, bias: bool, in_features: int, name: str | None, out_features: int
    ) -> None:
        # Kaiming uniform bound
        bound = 1.0 / np.sqrt(in_features) if in_features > 0 else 1.0
        w_val = np.random.uniform(-bound, bound, (out_features, in_features))
        w_name = f"{name}_weight" if name else "weight"
        self.weight = Parameter(w_val, name=w_name)

        if bias:
            b_val = np.random.uniform(-bound, bound, (out_features,))
            b_name = f"{name}_bias" if name else "bias"
            self.bias: Parameter | None = Parameter(b_val, name=b_name)
        else:
            self.bias = None

    def forward(self, x: ut.Tensor) -> ut.Tensor:
        """Forward linear transformation."""
        out = x @ self.weight.transpose()
        if self.bias is not None:
            out = out + self.bias
        return out


class ReLU(Module):
    """Applies element-wise Rectified Linear Unit activation: max(0, x)."""

    def forward(self, x: ut.Tensor) -> ut.Tensor:
        """Forward ReLU transformation."""
        return x.relu()


class Sigmoid(Module):
    """Applies element-wise Sigmoid activation: 1 / (1 + exp(-x))."""

    def forward(self, x: ut.Tensor) -> ut.Tensor:
        """Forward Sigmoid transformation."""
        return x.sigmoid()


class Sequential(Module):
    """Sequential container passing the output of each module into the next.

    Args:
        *layers: Variable sequence of Module instances.

    Examples:
        ```python
        import utorch.nn as nn

        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )
        ```
    """

    def __init__(self, *layers: Module) -> None:
        """Initialize a Sequential container."""
        super().__init__()
        self.layers = list(layers)

    def forward(self, x: Any) -> Any:
        """Forward pass through the sequential container."""
        for layer in self.layers:
            x = layer(x)
        return x

    def __getitem__(self, index: int) -> Module:
        """Return item at given index."""
        return self.layers[index]

    def __len__(self) -> int:
        """Return number of layers in sequential container."""
        return len(self.layers)
