from collections.abc import Callable

import numpy as np

from utorch import Tensor


def gradcheck(
    fun: Callable[[Tensor], Tensor],
    x_val: float,
    eps: float = 1e-6,
) -> tuple[float, float]:
    """Compare analytical autograd gradient against central finite differences.

    Formula:
        df/dx ≈ [f(x + eps) - f(x - eps)] / (2 * eps)
    """
    # 1. Analytical gradient via autograd
    x = Tensor(x_val)
    y = fun(x)
    y.backward()
    analytical = float(x.grad.value)

    # 2. Numerical approximation via two-sided finite difference
    f_plus = float(fun(Tensor(x_val + eps)).value)
    f_minus = float(fun(Tensor(x_val - eps)).value)
    numerical = (f_plus - f_minus) / (2.0 * eps)

    return analytical, numerical


def tensor_gradcheck(
    fn: Callable[..., Tensor],
    *inputs: Tensor,
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> bool:
    """Verify analytical gradients of a scalar-valued function fn(*inputs).

    Iterates through every single multi-dimensional coordinate across all input
    tensors, computing central finite differences and asserting numerical
    convergence with analytical autograd gradients.
    """
    # 1. Forward and analytical backward pass
    t_inputs = [Tensor(inp.value.copy()) for inp in inputs]
    loss = fn(*t_inputs)
    loss.backward()
    analytical_grads = [inp.grad.value.copy() for inp in t_inputs]

    # 2. Coordinate-by-coordinate numerical finite difference
    for inp_idx, inp in enumerate(inputs):
        num_grad = np.zeros_like(inp.value)
        it = np.nditer(inp.value, flags=["multi_index"])

        while not it.finished:
            idx = it.multi_index

            # Perturb +eps
            t_plus = [Tensor(x.value.copy()) for x in inputs]
            t_plus[inp_idx].value[idx] += eps
            loss_plus = float(fn(*t_plus).value)

            # Perturb -eps
            t_minus = [Tensor(x.value.copy()) for x in inputs]
            t_minus[inp_idx].value[idx] -= eps
            loss_minus = float(fn(*t_minus).value)

            num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
            it.iternext()

        ana_grad = analytical_grads[inp_idx]
        np.testing.assert_allclose(
            ana_grad,
            num_grad,
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradcheck failed for input {inp_idx} at shape {inp.shape}",
        )

    return True
