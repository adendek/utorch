"""Loss functions for training neural networks."""

from __future__ import annotations

import numpy as np

import utorch as ut
from utorch.nn.module import Module


class MSELoss(Module):
    """Mean Squared Error (L2) loss: L = mean((pred - target)^2)."""

    def forward(self, pred: ut.Tensor, target: ut.Tensor) -> ut.Tensor:
        """Compute mean squared error (L2) loss."""
        return ((pred - target) ** 2).mean()


class CrossEntropyLoss(Module):
    """Cross-Entropy loss with logits using numerically stable LogSumExp.

    Computes negative log-likelihood without requiring keepdims or ops changes:
        loss = - mean( sum( target * log_softmax(logits), axis=-1 ) )
    """

    def forward(self, logits: ut.Tensor, target: ut.Tensor) -> ut.Tensor:
        """Compute negative log-likelihood without requiring keepdims or ops changes."""
        # 1. Numerical stability: subtract max along class dimension
        # max_val has shape (64,)
        max_val = ut.Tensor(np.max(logits.value, axis=-1))
        # (64, 10).T - (64,) -> (10, 64).T -> (64, 10)
        shifted = (logits.transpose() - max_val).transpose()

        # 2. Sum along class axis (-1): produces shape (64,)
        sum_exp = shifted.exp().sum(axis=-1)
        log_sum_exp = sum_exp.log()  # Shape (64,)

        # 3. Transpose trick: (10, 64) - (64,) -> (10, 64).T -> (64, 10)
        log_probs = (shifted.transpose() - log_sum_exp).transpose()

        # 4. Negative log-likelihood against one-hot targets
        if target.shape == logits.shape:
            loss = (target * log_probs * (-1.0)).sum(axis=-1).mean()
        else:
            loss = log_probs.mean() * (-1.0)

        return loss
