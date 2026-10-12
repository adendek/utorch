"""Optimization algorithms for updating model parameters."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from utorch.nn.parameter import Parameter


class Optimizer:
    """Base class for all parameter optimizers."""

    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        """Initialize a parameter optimizer."""
        self.params: list[Parameter] = list(params)
        self.lr: float = lr

    def zero_grad(self) -> None:
        """Reset gradients for all tracked parameters."""
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """Perform a single parameter update step."""
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer with optional classical momentum.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate.
        momentum: Momentum factor (default: 0.0).
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-2,
        momentum: float = 0.0,
    ) -> None:
        """Initialize SGD optimizer."""
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = [np.zeros_like(p.value) for p in self.params]

    def step(self) -> None:
        """Perform a single parameter update step."""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad_val = p.grad.value

            if self.momentum > 0.0:
                self.velocities[i] = self.momentum * self.velocities[i] + grad_val
                p.value -= self.lr * self.velocities[i]
            else:
                p.value -= self.lr * grad_val
