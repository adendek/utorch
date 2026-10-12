import numpy as np

from tests.utils import tensor_gradcheck
from utorch import Tensor


def test_matmul_tensor_gradcheck() -> None:
    """Verify matrix multiplication VJPs across both operands simultaneously."""
    rng = np.random.default_rng(42)
    A = Tensor(rng.standard_normal((3, 4)))  # noqa: N806
    B = Tensor(rng.standard_normal((4, 2)))  # noqa: N806

    # Function returning scalar loss: sum(A @ B)
    assert tensor_gradcheck(lambda a, b: (a @ b).sum(), A, B)


def test_dense_layer_broadcasting_gradcheck() -> None:
    """Verify linear layer with 1D bias vector broadcasted across batch."""
    rng = np.random.default_rng(101)
    X = Tensor(rng.standard_normal((4, 3)))  # noqa: N806
    W = Tensor(rng.standard_normal((3, 2)))  # noqa: N806
    b = Tensor(rng.standard_normal(2))  # Shape (2,)

    # Function: sum(X @ W + b)
    assert tensor_gradcheck(lambda x, w, bias: ((x @ w) + bias).sum(), X, W, b)


def test_msle_loss_tensor_gradcheck() -> None:
    """Verify complex non-linear MSLE loss on weights and bias."""
    rng = np.random.default_rng(202)
    # Use positive numbers to stay within the real logarithm domain
    X = Tensor(np.abs(rng.standard_normal((4, 3))) + 0.1)  # noqa: N806
    W = Tensor(np.abs(rng.standard_normal((3, 2))) + 0.1)  # noqa: N806
    b = Tensor(np.abs(rng.standard_normal(2)) + 0.1)
    y_true = Tensor(np.abs(rng.standard_normal((4, 2))) + 0.5)

    def msle_objective(x: Tensor, w: Tensor, bias: Tensor) -> Tensor:
        preds = (x @ w) + bias
        # MSLE: mean( (log(1 + y) - log(1 + preds)) ** 2 )
        return (((1.0 + y_true).log() - (1.0 + preds).log()) ** 2).mean()

    assert tensor_gradcheck(msle_objective, X, W, b)


def test_axis_reductions_gradcheck() -> None:
    """Verify sum and mean backward across specific axes on 3D tensors."""
    rng = np.random.default_rng(303)
    # 3D Tensor: shape (2, 3, 4)
    T = Tensor(rng.standard_normal((2, 3, 4)))  # noqa: N806

    # Sum along axis 1
    assert tensor_gradcheck(lambda t: t.sum(axis=1).sum(), T)

    # Mean along multiple axes (0, 2)
    assert tensor_gradcheck(lambda t: t.mean(axis=(0, 2)).sum(), T)

    # Negative axis: sum along last dimension (-1)
    assert tensor_gradcheck(lambda t: t.sum(axis=-1).sum(), T)


def test_elementwise_broadcasting_gradcheck() -> None:
    """Verify elementwise multiplication where a row vector is broadcasted."""
    rng = np.random.default_rng(404)
    A = Tensor(rng.standard_normal((3, 4)))  # noqa: N806
    b = Tensor(rng.standard_normal(4))  # 1D vector of shape (4,)

    # A * b broadcasts across all 3 rows of A
    assert tensor_gradcheck(lambda a, vec: (a * vec).sum(), A, b)


def test_composite_activation_gradcheck() -> None:
    """Verify composite pipeline: Mean Squared Error on Sigmoid projection."""
    rng = np.random.default_rng(505)
    X = Tensor(rng.standard_normal((3, 2)))  # noqa: N806
    W = Tensor(rng.standard_normal((2, 2)))  # noqa: N806

    def pipeline(x: Tensor, w: Tensor) -> Tensor:
        return ((x @ w).sigmoid() ** 2).mean()

    assert tensor_gradcheck(pipeline, X, W)
