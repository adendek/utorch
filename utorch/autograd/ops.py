"""Scalar derivative rules for reverse-mode automatic differentiation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Binary ops: [d_left, d_right] where each is (grad, left, right, args) -> float
# Unary ops: d_parent where function is (grad, parent, args) -> float
ScalarRule = list[Callable[..., Any]] | Callable[..., Any]

SCALAR_PRIMITIVES: dict[str, ScalarRule] = {
    # Addition: d/dx(x + y) = 1, d/dy(x + y) = 1
    "__add__": [
        lambda grad, left, right, args: grad,
        lambda grad, left, right, args: grad,
    ],
    # Right side addition: d/dx(x + y) = 1, d/dy(x + y) = 1
    "__radd__": [
        lambda grad, left, right, args: grad,
        lambda grad, left, right, args: grad,
    ],
    # Subtraction: d/dx(x - y) = 1, d/dy(x - y) = -1
    "__sub__": [
        lambda grad, left, right, args: grad,
        lambda grad, left, right, args: -grad,
    ],
    # Right side subtraction: d/dx(y - x) = -1, d/dy(y -x) = 1
    "__rsub__": [
        lambda grad, left, right, args: -grad,
        lambda grad, left, right, args: grad,
    ],
    # Multiplication: d/dx(x * y) = y, d/dy(x * y) = x
    "__mul__": [
        lambda grad, left, right, args: grad * right.value,
        lambda grad, left, right, args: grad * left.value,
    ],
    # Division: d/dx(x / y) = 1/y, d/dy(x / y) = -x / y^2
    "__truediv__": [
        lambda grad, left, right, args: grad / right.value,
        lambda grad, left, right, args: grad * (-1) * left.value / (right.value**2),
    ],
    # Power: d/dx(x^p) = p * x^(p-1)
    "__pow__": lambda grad, x, p: grad * p * (x ** (p - 1)),
    # Unary Activations & Trigonometry
    "neg": lambda grad, x, args: grad * (-1.0),
    "sin": lambda grad, x, args: grad * x.cos(),
    "cos": lambda grad, x, args: grad * (-1.0) * x.sin(),
    "exp": lambda grad, x, args: grad * x.exp(),
    "log": lambda grad, x, args: grad / x,
    "tanh": lambda grad, x, args: grad * (1.0 - (x.tanh() ** 2)),
    "sigmoid": lambda grad, x, args: grad * x.sigmoid() * (1.0 - x.sigmoid()),
    "relu": lambda grad, x, args: grad * (1.0 if x.value > 0.0 else 0.0),
}
