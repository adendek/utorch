from __future__ import annotations

import numpy as np

import utorch as ut


def unbroadcast(grad: ut.Tensor, target_shape: tuple[int, ...]) -> ut.Tensor:
    """Project a broadcasted gradient tensor back to its input operand shape.

    During forward evaluation of binary operations (e.g., A + B), smaller
    operands are implicitly expanded via NumPy broadcasting rules (e.g., a bias
    vector of shape (10,) added to a batch matrix of shape (32, 10)).

    In the backward pass, incoming gradients match the expanded output shape
    (32, 10). To maintain dimensional alignment, the gradient must be summed
    across all broadcasted axes so that grad.shape == target_shape.

    Args:
        grad: Incoming gradient Tensor with the expanded shape.
        target_shape: The original shape of the parent input Tensor.

    Returns:
        A new Tensor reduced to match `target_shape`.

    Examples:
        ```python
        import numpy as np
        import utorch as ut
        from utorch.autograd.ops import unbroadcast

        # Batch gradient of shape (32, 10)
        grad = ut.Tensor(np.ones((32, 10)))

        # Unbroadcast to bias vector shape (10,)
        bias_grad = unbroadcast(grad, (10,))
        print(bias_grad.shape)  # (10,)
        print(bias_grad.value[0])  # 32.0 (summed across batch dimension 0)
        ```
    """
    grad_val = grad.value
    if grad_val.shape == target_shape:
        return grad

    grad_ndim = grad_val.ndim
    target_ndim = len(target_shape)

    # 1. Sum over extra leading dimensions (e.g., (32, 10) -> (10,))
    if grad_ndim > target_ndim:
        leading_axes = tuple(range(grad_ndim - target_ndim))
        grad_val = np.sum(grad_val, axis=leading_axes)

    # 2. Sum over size-1 dimensions where broadcasting expanded size > 1
    axes_to_keep: list[int] = []
    for i, (g_dim, t_dim) in enumerate(zip(grad_val.shape, target_shape, strict=False)):
        if t_dim == 1 and g_dim > 1:
            axes_to_keep.append(i)

    if axes_to_keep:
        grad_val = np.sum(grad_val, axis=tuple(axes_to_keep), keepdims=True)

    return ut.Tensor(grad_val.reshape(target_shape))
