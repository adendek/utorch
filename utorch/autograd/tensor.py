"""Computational graph node and tensor abstraction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np


class Tensor:
    """A multidimensional array and computational graph node.

    `Tensor` tracks numerical data alongside the operations and parent nodes
    that produced it, dynamically building a Directed Acyclic Graph (DAG) during
    the forward pass.

    Args:
        value: Numerical data (scalar, list, or NumPy ndarray). If None,
            the node is initialized as an intermediate operation node.
        parents: Upstream antecedent nodes that fed into this computation.
        fun: The forward mathematical function evaluated at this node.
        name: Optional human-readable identifier for visualization.
        args: Optional extra static arguments passed to `fun` (e.g. axes, min/max).

    Attributes:
        value (np.ndarray): The evaluated forward output tensor.
        parents (list[Tensor]): Direct input nodes in the computational graph.
        fun (Callable | None): The mathematical operation that generated this node.
        args (Any): Additional constants or axes supplied to the operation.
        name (str | None): Label used in graph diagrams and debugging.
        grad (int | float | np.ndarray): Accumulated gradient (computed in Branch 3+).

    Example:
        ```python
        import utorch as ut

        x = ut.Tensor(2.0, name="x")
        y = ut.Tensor(3.0, name="y")
        z = x * y + 1.0

        print(z.value)   # 7.0
        print(z.is_leaf) # False
        ```
    """

    def __init__(
        self,
        value: Any = None,
        parents: Sequence[Tensor] | None = None,
        fun: Callable[..., Any] | None = None,
        name: str | None = None,
        args: Any = None,
    ) -> None:
        r"""Initialize a computational graph node via leaf or inner node dispatch.

        In a dynamic Directed Acyclic Graph (DAG), every node belongs to one of
        two fundamental categories:

        1. **Leaf Nodes (Inputs, Constants, Parameters)**:
           - Created directly by the user with a known starting `value`.
           - Have **no antecedent parents** (`parents = []`) and **no creator
             function** (`fun = None`).
           - Represent independent variables at the base of the tree (e.g.,
             weights, biases, inputs).
           - Dispatched to: `_init_if_leaf(value)`.

        2. **Inner Nodes (Intermediate Operations)**:
           - Created automatically as the output of an operation (e.g., `+`,
             `*`, `relu`).
           - Have **antecedent parent nodes** (`parents`) and an **associated
             mathematical function** (`fun`).
           - Numerical value is evaluated dynamically during the forward pass by
             applying `fun(*parent_values)`.
           - Dispatched to: `_init_as_inner_node(parents, fun, args)`.

        Args:
            value: Numerical data (scalar int/float, list, or NumPy array).
                If provided, initialized as a **leaf node**.
            parents: Upstream antecedent `Tensor` nodes that fed into this
                operation. Used when initializing an **inner node**.
            fun: The forward mathematical function evaluated at this node
                (e.g., `np.add`, `np.sin`). Required for inner nodes; must be
                `None` for leaf nodes.
            name: Optional human-readable identifier for visualization.
            args: Optional static constants passed to `fun` (e.g., reduction
                axis, power exponent).

        """
        self.name: str | None = name
        if value is not None:
            self._init_if_leaf(value)
        else:
            self._init_as_inner_node(parents or [], fun, args)
        if value is not None:
            self._init_if_leaf(value)
        else:
            self._init_as_inner_node(parents or [], fun, args)

        if value is not None:
            self._init_if_leaf(value)
        else:
            self._init_as_inner_node(parents or [], fun, args)

    def _init_if_leaf(self, value: Any) -> None:
        """Initialize this variable as a user-supplied leaf node (no parents)."""
        if isinstance(value, np.ndarray):
            self.value = value
        elif isinstance(value, int | float | bool):
            self.value = np.array(value, dtype=np.float64)
        else:
            self.value = np.asarray(value, dtype=np.float64)

        self._is_leaf: bool = True
        self.fun: Callable[..., Any] | None = None
        self.args: Any = None
        self.parents: list[Tensor] = []

    def _init_as_inner_node(
        self,
        parents: Sequence[Tensor],
        fun: Callable[..., Any] | None,
        args: Any = None,
    ) -> None:
        """Initialize this variable as an intermediate operation node."""
        if fun is None:
            raise ValueError(
                "Inner computational graph nodes must specify a function 'fun'."
            )

        self.parents = list(parents)
        parent_values = [p.value for p in self.parents]

        # Execute forward function pass
        if args is None:
            self.value = fun(*parent_values)
        else:
            self.value = fun(*parent_values, args)

        self.fun = fun
        self.args = args
        self._is_leaf = False

    @property
    def is_leaf(self) -> bool:
        """Return True if this node has no parents (user input or parameter)."""
        return self._is_leaf

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the dimensional shape of the underlying tensor."""
        if isinstance(self.value, int | float):
            return (1,)
        return self.value.shape  # type: ignore[no-any-return]

    @classmethod
    def _ensure_tensor(cls, other: Any) -> Tensor:
        """Lift numeric constants or ndarrays into a Variable if not already one.

        Args:
            other: Any numeric value, list, array, or existing Variable.

        Returns:
            A valid Variable instance.
        """
        if isinstance(other, Tensor):
            return other
        return cls(value=other)

    def single_operator(self, fun: Callable[..., Any], args: Any = None) -> Tensor:
        """Apply a unary operation taking self as the sole parent node.

        Args:
            fun: Unary mathematical function (e.g. np.sin, np.exp).
            args: Optional static parameters.

        Returns:
            A new Variable representing the operation's output.
        """
        return Tensor(parents=[self], fun=fun, args=args)

    def _two_variable_operator(
        self,
        other: Any,
        fun: Callable[..., Any],
        args: Any = None,
    ) -> Tensor:
        """Apply a binary operation taking self and other as parent nodes.

        Args:
            other: Second operand (automatically lifted to Variable).
            fun: Binary mathematical function (e.g. np.add, np.multiply).
            args: Optional static parameters.

        Returns:
            A new Variable representing the operation's output.
        """
        other_var = self._ensure_tensor(other)
        return Tensor(parents=[self, other_var], fun=fun, args=args)

    # =========================================================================
    # Arithmetic Operator Overloads
    # =========================================================================

    def __add__(self, other: Any) -> Tensor:
        """Element-wise addition: `self + other`."""
        return self._two_variable_operator(other, lambda x, y: np.add(x, y))

    def __radd__(self, other: Any) -> Tensor:
        """Right-hand element-wise addition: `other + self`."""
        return self.__add__(other)

    def __sub__(self, other: Any) -> Tensor:
        """Element-wise subtraction: `self - other`."""
        return self._two_variable_operator(other, lambda x, y: np.subtract(x, y))

    def __rsub__(self, other: Any) -> Tensor:
        """Right-hand element-wise subtraction: `other - self`."""
        other_var = self._ensure_tensor(other)
        return other_var._two_variable_operator(self, lambda x, y: np.subtract(x, y))

    def __mul__(self, other: Any) -> Tensor:
        """Element-wise multiplication: `self * other`."""
        return self._two_variable_operator(other, lambda x, y: np.multiply(x, y))

    def __rmul__(self, other: Any) -> Tensor:
        """Right-hand element-wise multiplication: `other * self`."""
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> Tensor:
        """Element-wise division: `self / other`."""
        return self._two_variable_operator(other, lambda x, y: np.divide(x, y))

    def __rtruediv__(self, other: Any) -> Tensor:
        """Right-hand element-wise division: `other / self`."""
        other_var = self._ensure_tensor(other)
        return other_var._two_variable_operator(self, lambda x, y: np.divide(x, y))

    def __matmul__(self, other: Any) -> Tensor:
        """Matrix multiplication: `self @ other`."""
        return self._two_variable_operator(other, lambda x, y: np.matmul(x, y))

    def __pow__(self, exponent: int | float) -> Tensor:
        """Element-wise power: `self ** exponent`."""
        return self.single_operator(lambda x, p: np.power(x, p), args=exponent)

    def __neg__(self) -> Tensor:
        """Element-wise negation: `-self`."""
        return self.single_operator(lambda x: np.negative(x))

    # =========================================================================
    # Comparison Operators (Evaluated numerically on values)
    # =========================================================================

    def __ge__(self, other: Any) -> Any:
        """Element-wise greater or equal: `self >= other`."""
        other_val = other.value if isinstance(other, Tensor) else other
        return np.greater_equal(self.value, other_val)

    def __le__(self, other: Any) -> Any:
        """Element-wise less or equal: `self <= other`."""
        other_val = other.value if isinstance(other, Tensor) else other
        return np.less_equal(self.value, other_val)

    def __eq__(self, other: Any) -> Any:
        """Element-wise equal: `self == other`."""
        if not isinstance(other, Tensor):
            return False
        return np.array_equal(self.value, other.value)

    def __lt__(self, other: Any) -> Any:
        """Element-wise less than: `self < other`."""
        other_val = other.value if isinstance(other, Tensor) else other
        return np.less(self.value, other_val)

    def __gt__(self, other: Any) -> Any:
        """Element-wise greater than: `self > other`."""
        other_val = other.value if isinstance(other, Tensor) else other
        return np.greater(self.value, other_val)

    def __hash__(self) -> int:
        """Calculates hash(value)`."""
        return hash(id(self))

    # =========================================================================
    # Unary Mathematical Operations & Activations
    # =========================================================================

    def exp(self) -> Tensor:
        r"""Element-wise exponential: $\\exp(x)$."""
        return self.single_operator(lambda x: np.exp(x))

    def log(self) -> Tensor:
        r"""Element-wise natural logarithm: $\\ln(x)$."""
        return self.single_operator(lambda x: np.log(x))

    def sin(self) -> Tensor:
        r"""Element-wise sine: $\\sin(x)$."""
        return self.single_operator(lambda x: np.sin(x))

    def cos(self) -> Tensor:
        r"""Element-wise cosine: $\\cos(x)$."""
        return self.single_operator(lambda x: np.cos(x))

    def tanh(self) -> Tensor:
        r"""Element-wise hyperbolic tangent: $\\tanh(x)$."""
        return self.single_operator(lambda x: np.tanh(x))

    def cosh(self) -> Tensor:
        r"""Element-wise hyperbolic cosine: $\\cosh(x)$."""
        return self.single_operator(lambda x: np.cosh(x))

    def abs(self) -> Tensor:
        r"""Element-wise absolute value: $|x|$."""
        return self.single_operator(lambda x: np.abs(x))

    def sign(self) -> Tensor:
        r"""Element-wise sign indicator (-1, 0, or 1)."""
        return self.single_operator(lambda x: np.sign(x))

    def relu(self) -> Tensor:
        r"""Rectified Linear Unit activation: $\\max(0, x)$."""
        return self.single_operator(lambda x: np.maximum(0, x))

    def sigmoid(self) -> Tensor:
        r"""Sigmoid activation function: $\\sigma(x) = \\frac{1}{1 + e^{-x}}$."""
        return self.single_operator(lambda x: 1.0 / (1.0 + np.exp(-x)))

    # Reductions & Shape Operations
    def sum(self, axis: int | tuple[int, ...] | None = None) -> Tensor:
        """Sum tensor elements across specified axis."""
        return self.single_operator(lambda x, ax: np.sum(x, axis=ax), args=axis)

    def mean(self, axis: int | tuple[int, ...] | None = None) -> Tensor:
        """Compute the arithmetic mean of tensor elements across specified axis."""
        return self.single_operator(lambda x, ax: np.mean(x, axis=ax), args=axis)

    def transpose(self, axes: Sequence[int] | None = None) -> Tensor:
        """Permute the dimensions of the tensor."""
        return self.single_operator(lambda x, ax: np.transpose(x, axes=ax), args=axes)

    def clip_min(self, min_val: float) -> Tensor:
        """Clip tensor values below a minimum threshold."""
        return self.single_operator(
            lambda x, m: np.clip(x, a_min=m, a_max=None), args=min_val
        )

    # =========================================================================
    # Representation & Placeholder
    # =========================================================================

    def backward(self) -> None:
        """Placeholder for reverse-mode automatic differentiation.

        Introduced in Branch 3 (scalar) and Branch 4 (tensor vector-Jacobian products).
        """
        raise NotImplementedError(
            "Backward pass is not yet available in Branch 2 (Forward graphs only). "
            "Switch to Branch 3 (feat/03-scalar-autograd) for autograd."
        )

    def __repr__(self) -> str:
        """Return a string representation of the tensor for inspection."""
        name_str = f"'{self.name}', " if self.name else ""
        if self.is_leaf:
            return f"ut.Tensor({name_str}value={self.value})"
        fun_name = getattr(self.fun, "__name__", "op")
        return f"ut,.Tensor({name_str}value={self.value}, fun={fun_name})"
