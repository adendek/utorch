from collections.abc import Callable

import numpy as np
import pytest

from tests.utils import gradcheck
from utorch import Tensor


@pytest.mark.parametrize(
    "name, fun, test_val",
    [
        ("square", lambda x: x**2, 3.0),
        ("cube", lambda x: x**3, -2.5),
        ("fractional_pow", lambda x: x**1.5, 4.0),
        ("sin", lambda x: x.sin(), 1.2),
        ("cos", lambda x: x.cos(), 0.8),
        ("exp", lambda x: x.exp(), 0.5),
        ("log", lambda x: x.log(), 2.5),
        ("tanh", lambda x: x.tanh(), -0.7),
        ("sigmoid", lambda x: x.sigmoid(), 1.5),
        ("composite_trig", lambda x: (x**2).sin() + 3.0 * x, 1.1),
        ("rational", lambda x: (x + 1.0) / (x - 1.0), 3.0),
    ],
)
def test_primitive_derivatives_gradcheck(
    name: str,
    fun: Callable[[Tensor], Tensor],
    test_val: float,
) -> None:
    """Verify analytical gradient matches numerical finite difference."""
    analytical, numerical = gradcheck(fun, test_val)
    np.testing.assert_allclose(
        analytical,
        numerical,
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Gradcheck failed for {name}: "
        f"analytical={analytical},"
        f" numerical={numerical}",
    )


def test_relu_gradient() -> None:
    """Verify ReLU subgradient: 1.0 for positive x, 0.0 for negative x."""
    x_pos = Tensor(2.5)
    (x_pos.relu()).backward()
    assert np.isclose(x_pos.grad.value, 1.0)

    x_neg = Tensor(-2.5)
    (x_neg.relu()).backward()
    assert np.isclose(x_neg.grad.value, 0.0)


def test_diamond_graph_addition_accumulation() -> None:
    """Verify y = x + x accumulates gradients to 2.0 (not overwriting to 1.0)."""
    x = Tensor(3.0, name="x")
    y = x + x  # dy/dx = 1 + 1 = 2
    y.backward()
    assert np.isclose(x.grad.value, 2.0)


def test_diamond_graph_product_rule() -> None:
    """Verify product rule via autograd: y = x * x -> dy/dx = 2x."""
    x = Tensor(4.0, name="x")
    y = x * x  # dy/dx = 2 * 4 = 8.0
    y.backward()
    assert np.isclose(x.grad.value, 8.0)


def test_polynomial_multi_path_accumulation() -> None:
    """Verify multi-branch polynomial: y = x^3 - 2x^2 + 5x - 7."""
    # dy/dx = 3x^2 - 4x + 5
    # At x = 2.0: 3*(4) - 4*(2) + 5 = 12 - 8 + 5 = 9.0
    x = Tensor(2.0, name="x")
    y = (x**3) - 2.0 * (x**2) + 5.0 * x - 7.0
    y.backward()
    assert np.isclose(x.grad.value, 9.0)


def test_second_derivative_cubic_polynomial() -> None:
    """Verify double backward computes exact second derivative d2y/dx2.

    y = x^3
    dy/dx = 3x^2   -> at x=2: 12.0
    d2y/dx2 = 6x   -> at x=2: 12.0
    """
    x = Tensor(2.0, name="x")
    y = x**3

    # Pass 1: First derivative
    y.backward()
    assert np.isclose(x.grad.value, 12.0)

    # Pass 2: Backpropagate THROUGH the first gradient!
    first_grad = x.grad
    x.zero_grad()
    first_grad.backward()
    assert np.isclose(x.grad.value, 12.0)


def test_cyclic_trigonometric_higher_order_derivatives() -> None:
    """Verify cyclic derivatives: sin -> cos -> -sin -> -cos.

    At x = 0.0:
    f(0)   = sin(0)   = 0.0
    f'(0)  = cos(0)   = 1.0
    f''(0) = -sin(0)  = 0.0
    f'''(0)= -cos(0)  = -1.0
    """
    x = Tensor(0.0, name="x")
    y = x.sin()

    # f'(0) = 1.0
    y.backward()
    assert np.isclose(x.grad.value, 1.0)

    # f''(0) = 0.0
    g1 = x.grad
    x.zero_grad()
    g1.backward()
    assert np.isclose(x.grad.value, 0.0)

    # f'''(0) = -1.0
    g2 = x.grad
    x.zero_grad()
    g2.backward()
    assert np.isclose(x.grad.value, -1.0)


def test_linear_second_derivative_evaluates_to_zero() -> None:
    """Verify that the second derivative of a linear function evaluates to 0.0."""
    x = Tensor(5.0, name="x")
    y = 3.0 * x + 7.0  # dy/dx = 3.0, d2y/dx2 = 0.0

    y.backward()
    assert np.isclose(x.grad.value, 3.0)

    # Second backward: x is not in the lineage of the constant 3.0
    g1 = x.grad
    x.zero_grad()
    g1.backward()

    # Should remain 0.0 or None (disconnected graph)
    second_deriv = 0.0 if x.grad is None else float(x.grad.value)
    assert np.isclose(second_deriv, 0.0)
