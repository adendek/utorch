"""Scalar derivative rules for reverse-mode automatic differentiation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from utorch.autograd.tensor import Tensor


def sum_backward(
    grad: Tensor,
    x: Tensor,
    axis: int | tuple[int, ...] | None = None,
) -> Tensor:
    """Compute the Vector-Jacobian Product (VJP) for tensor summation.

    Mathematical Intuition:
        Summation distributes incoming gradients equally across all elements
        that contributed to the sum. For Y = sum(X), dY/dX_i = 1.0.
        Therefore: dL/dX = dL/dY * 1.0 (broadcasted to X's original shape).

    Broadcasting Mechanics:
        When an axis is collapsed during sum(), that dimension is eliminated
        from the output shape (e.g., shape (2, 4, 3) reduced along axis=1
        becomes shape (2, 3)).

        To broadcast the incoming gradient back to (2, 4, 3), we must:
        1. Normalize negative indices (e.g., axis=-1 becomes axis=2).
        2. Re-insert a singleton dimension (size 1) at the collapsed axes:
           (2, 3) -> reshape -> (2, 1, 3).
        3. Use np.broadcast_to to stretch size 1 back to the original size 4:
           (2, 1, 3) -> broadcast -> (2, 4, 3).
    """
    x_shape = x.shape

    # Scenario 1: Total sum across all elements (axis is None).
    # The output was a scalar; every element in X receives the incoming gradient.
    if axis is None:
        return Tensor(np.broadcast_to(grad.value, x_shape).copy())

    # Scenario 2: Sum along specific axis or axes.
    # Step A: Normalize input axis into a tuple of positive dimension indices.
    # (e.g., for a 3D tensor, axis=-1 is converted to 3 + (-1) = 2).
    axes = (axis,) if isinstance(axis, int) else tuple(axis)
    pos_axes = tuple(a if a >= 0 else len(x_shape) + a for a in axes)

    # Step B: Reconstruct shape with singleton dimensions (1) at reduced positions.
    # Example: If x_shape is (2, 4, 3) and pos_axes is (1,),
    # expanded_shape becomes [2, 1, 3].
    expanded_shape = list(x_shape)
    for a in pos_axes:
        expanded_shape[a] = 1

    # Step C: Reshape gradient to include the singleton dimensions, then stretch.
    # grad.value (2, 3) -> reshaped to (2, 1, 3) -> broadcasted to (2, 4, 3).
    grad_reshaped = np.reshape(grad.value, expanded_shape)
    return Tensor(np.broadcast_to(grad_reshaped, x_shape).copy())


def mean_backward(
    grad: Tensor,
    x: Tensor,
    axis: int | tuple[int, ...] | None = None,
) -> Tensor:
    """Compute the Vector-Jacobian Product (VJP) for tensor mean reduction.

    Mathematical Intuition:
        The arithmetic mean is a scaled summation:
            mean(X) = (1 / N) * sum(X)
        where N is the number of elements collapsed along the reduced axes.

        By the chain rule, every element that contributed to the mean receives
        an equal fractional share of the incoming gradient:
            dL/dX_i = (1 / N) * dL/dY

        Therefore, mean_backward is simply sum_backward scaled by (1 / N):
            dL/dX = (1 / N) * sum_backward(grad, x, axis)

    Broadcasting & Normalization Mechanics:
        1. Determine N (the count of elements reduced):
           - If axis is None: N is the total count of all elements in X.
           - If axis is specified (e.g. axis=0 on shape (4, 3)): N is the
             product of the lengths of the reduced dimensions (here N = 4).
        2. Leverage sum_backward to re-insert singleton dimensions and stretch
           the gradient back to x.shape.
        3. Divide the broadcasted gradient by N.

    Args:
        grad: Incoming gradient Tensor from downstream nodes.
        x: The parent Tensor that was reduced.
        axis: The axis or axes along which the mean was computed.

    Returns:
        Tensor of shape `x.shape` scaled by the inverse element count (1 / N).
    """
    x_shape = x.shape

    # Step 1: Calculate the normalization factor N (number of collapsed elements).
    if axis is None:
        # Total mean: all elements in the entire array were averaged together.
        num_elements = int(np.prod(x_shape)) if x_shape else 1
    else:
        # Axis-specific mean: count elements along the specified reduced dimensions.
        axes = (axis,) if isinstance(axis, int) else tuple(axis)
        pos_axes = tuple(a if a >= 0 else len(x_shape) + a for a in axes)
        num_elements = int(np.prod([x_shape[a] for a in pos_axes]))

    # Step 2: Use sum_backward to broadcast grad back to the original x_shape.
    grad_sum = sum_backward(grad, x, axis)

    # Step 3: Scale by 1 / N so each contributing element receives its share.
    return Tensor(grad_sum.value / float(num_elements))


def matmul_backward(
    grad: Tensor,
    left: Tensor,
    right: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute the Vector-Jacobian Product (VJP) for matrix multiplication.

    Mathematical Intuition & Derivation:
        Let Y = A @ B, where:
            A has shape (M, K)
            B has shape (K, N)
            Y has shape (M, N)
        Let G = dL/dY be the incoming gradient of shape (M, N).

        Instead of computing the massive (M*N) x (M*K) Jacobian matrix,
        reverse-mode autograd evaluates the Vector-Jacobian Product directly.

        By index notation, each output element is:
            Y_{i, j} = sum_k (A_{i, k} * B_{k, j})

        Taking partial derivatives with respect to an entry A_{i, k}:
            dL/dA_{i, k} = sum_j (dL/dY_{i, j} * dY_{i, j}/dA_{i, k})
                         = sum_j (G_{i, j} * B_{k, j})
                         = sum_j (G_{i, j} * (B^T)_{j, k})
        In matrix form:
            dL/dA = G @ B^T   (Shape: (M, N) @ (N, K) -> (M, K))

        Similarly, for an entry B_{k, j}:
            dL/dB_{k, j} = sum_i (dL/dY_{i, j} * dY_{i, j}/dB_{k, j})
                         = sum_i (G_{i, j} * A_{i, k})
                         = sum_i ((A^T)_{k, i} * G_{i, j})
        In matrix form:
            dL/dB = A^T @ G   (Shape: (K, M) @ (M, N) -> (K, N))

    Dimensional Sanity Check (The "Shape Alignment Rule"):
        - dL/dA must match A.shape (M, K):
          G (M, N) @ B.T (N, K) -> (M, K)  [Inner dimension N cancels]
        - dL/dB must match B.shape (K, N):
          A.T (K, M) @ G (M, N) -> (K, N)  [Inner dimension M cancels]

    Args:
        grad: Incoming gradient Tensor G of shape (M, N).
        left: Matrix A of shape (M, K).
        right: Matrix B of shape (K, N).

    Returns:
        Tuple of (dL/dA, dL/dB) matching shapes of A and B respectively.
    """

    g_val = grad.value
    l_val = left.value
    r_val = right.value

    # Case 1: Dot product of two 1D vectors (output is a 0-D scalar)
    # y = u @ v -> dy/du = v, dy/dv = u
    if g_val.ndim == 0 or (l_val.ndim == 1 and r_val.ndim == 1):
        d_left = grad * right
        d_right = grad * left
        return d_left, d_right

    # Case 2: Matrix @ 1D vector: (B, D) @ (D,) -> (B,)
    if l_val.ndim == 2 and r_val.ndim == 1:
        # dL/dX = outer product G @ w^T: (B, 1) * (1, D) -> (B, D)
        g_2d = Tensor(g_val.reshape(-1, 1))
        r_2d = Tensor(r_val.reshape(1, -1))
        d_left = g_2d @ r_2d
        # dL/dw = X^T @ G: (D, B) @ (B,) -> (D,)
        d_right = left.transpose() @ grad
        return d_left, d_right

    # Case 3: Standard 2D or batched matrix multiplication
    d_left = grad @ right.transpose()
    d_right = left.transpose() @ grad
    return d_left, d_right


# Binary ops: [d_left, d_right] where each is (grad, left, right, args) -> float
# Unary ops: d_parent where function is (grad, parent, args) -> float
ScalarRule = list[Callable[..., Any]] | Callable[..., Any]


PRIMITIVES: dict[str, ScalarRule] = {
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
    "relu": lambda grad, x, args: grad * Tensor(np.where(x.value > 0.0, 1.0, 0.0)),
    # Tensor and  Linear Algebra & Reductions
    "__matmul__": lambda grad, left, right, args: matmul_backward(grad, left, right),
    "sum": lambda grad, x, axis: sum_backward(grad, x, axis),
    "mean": lambda grad, x, axis: mean_backward(grad, x, axis),
    "transpose": lambda grad, x, axes: (
        grad.transpose()
        if axes is None
        else Tensor(np.transpose(grad.value, np.argsort(axes)))
    ),
}
