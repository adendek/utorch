import numpy as np

from utorch import Tensor
from utorch.utils import count_nodes


def test_leaf_and_arithmetic_graph() -> None:
    """Test with leaf and arithmetic graphs."""
    x = Tensor(2.0, name="x")
    w = Tensor(3.0, name="w")
    b = Tensor(1.5, name="b")

    # (2 * 3) + 1.5 = 7.5
    y = x * w + b

    assert x.is_leaf
    assert w.is_leaf
    assert b.is_leaf
    assert not y.is_leaf
    assert np.isclose(y.value, 7.5)
    assert len(y.parents) == 2

    stats = count_nodes(y)
    assert stats.leaves == 3  # x, x, b
    assert stats.interim == 2  # '*', '+'
    assert stats.total == 5
    assert stats.edges == 4  # x->*, w->*, *->+, b->+


def test_mixed_scalar_and_unary_ops() -> None:
    """Test with scalar and unary ops."""
    x = Tensor(0.0)
    # sin(0) + 10 = 10.0
    res = x.sin() + 10
    assert np.isclose(res.value, 10.0)


def test_matrix_multiplication() -> None:
    """Test matrix multiplication."""
    a = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]))
    b = Tensor(np.array([[2.0, 0.0], [1.0, 2.0]]))
    c = a @ b

    expected = np.array([[4.0, 4.0], [10.0, 8.0]])
    assert np.allclose(c.value, expected)


def test_algebraic_identities() -> None:
    """Verify core algebraic invariants hold inside computation graphs."""
    rng = np.random.default_rng(1337)
    arr = rng.uniform(0.5, 5.0, size=(4, 4))
    x = Tensor(arr)

    # 1. Identity elements: x + 0 == x and x * 1 == x
    np.testing.assert_allclose((x + 0.0).value, arr)
    np.testing.assert_allclose((x * 1.0).value, arr)

    # 2. Commutativity: a + b == b + a and a * b == b * a
    y = Tensor(rng.uniform(0.5, 5.0, size=(4, 4)))
    np.testing.assert_allclose((x + y).value, (y + x).value)
    np.testing.assert_allclose((x * y).value, (y * x).value)

    # 3. Inverse operations: exp(log(x)) == x
    np.testing.assert_allclose(x.log().exp().value, arr, rtol=1e-5)

    # 4. Trigonometric identity: sin^2(x) + cos^2(x) == 1
    trig_identity = (x.sin() ** 2) + (x.cos() ** 2)
    np.testing.assert_allclose(trig_identity.value, np.ones_like(arr), rtol=1e-5)


def test_diamond_graph_topology() -> None:
    """Assert multi-path dependencies (diamond graphs) are accurately recorded.

    Diamond Graph Topology:
            x
           / \
          y1  y2
           |  /
            z
    """
    x = Tensor(2.0, name="x")
    y1 = x * 3.0  # 6.0
    y2 = x + 5.0  # 7.0
    z = y1 + y2  # 13.0

    assert z.value == 13.0
    assert len(z.parents) == 2
    assert z.parents[0] is y1
    assert z.parents[1] is y2

    # Both y1 and y2 must share the identical upstream parent node 'x'
    assert y1.parents[0] is x
    assert y2.parents[0] is x

    stats = count_nodes(z)
    # Leaves: x, 2.0, 3.0 -> 3 unique leaves
    assert stats.leaves == 3
    # Interim: y1 (*), y2 (+), z (+) -> 3 interim operations
    assert stats.interim == 3
    assert stats.total == 6
