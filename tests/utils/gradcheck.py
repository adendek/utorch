from collections.abc import Callable

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
