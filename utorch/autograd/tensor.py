"""Computational graph node and tensor abstraction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from utorch.autograd.ops import SCALAR_PRIMITIVES
from utorch.autograd.topological_sort import topological_sort


def _get_function_name(fun: Callable[..., Any]) -> str:
    """Extract a clean identifier string from a function or callable object.

    Parses the function's string representation or `__name__` attribute to
    retrieve the operation key used for looking up analytical derivative
    rules in the primitive dispatch dictionary.

    Args:
        fun: Callable mathematical function or ufunc (e.g., `np.sin`, `Tensor.__add__`).

    Returns:
        The extracted operation name string (e.g., "sin", "add", "cos").
    """
    if hasattr(fun, "__name__") and fun.__name__ != "<lambda>":
        return str(fun.__name__)
    return str(fun).split(".")[1]


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

        self.grad = None

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

    def __repr__(self) -> str:
        """Return a string representation of the tensor for inspection."""
        name_str = f"'{self.name}', " if self.name else ""
        if self.is_leaf:
            return f"ut.Tensor({name_str}value={self.value})"
        return (
            f"ut,.Tensor({name_str}value={self.value}, "
            f"fun={_get_function_name(self.fun)})"  # type: ignore[arg-type]
        )

    def zero_grad(self) -> None:
        """Reset gradient to zero on this node."""
        self.grad = None

    def backward(self, grad_tensor: Tensor | None = None) -> None:
        """Execute reverse-mode automatic differentiation (backpropagation).

        Computes the gradient of this output tensor with respect to all upstream
        nodes in the computational graph using the multivariate chain rule.

        Because gradients are accumulated as `Tensor` instances rather than raw
        floats, the backward pass dynamically constructs a secondary computation
        graph, enabling arbitrary higher-order derivatives (e.g., double backward
        for Hessians).

        The backward algorithm executes in five distinct pedagogical phases:

        1. **Scalar Verification & Seed Initialization**:
           - Reverse-mode autograd starts at a scalar objective (e.g., loss).
           - The root gradient is seeded with the identity: dL/dL = 1.0.
           - If an explicit `grad_tensor` is supplied, it is used as the seed.

        2. **Reverse Topological Ordering**:
           - Performs a Depth-First Search (DFS) from `self` backwards to all
             reachable ancestor nodes.
           - Arranges nodes so every child appears strictly before its parents.
           - Guarantees that when a node is processed, it has already collected
             all incoming gradient contributions from all downstream paths.

        3. **Primitive Derivative Dispatch**:
           - Traverses nodes in reverse topological order.
           - For each intermediate operation node, looks up its analytical
             derivative rule in `SCALAR_PRIMITIVES` using `node.op`.

        4. **Local Chain Rule Evaluation**:
           - For binary operations z = f(x, y):
             dx = d_rule_left(grad_z, x, y)
             dy = d_rule_right(grad_z, x, y)
           - For unary operations y = f(x):
             dx = d_rule(grad_y, x)

        5. **Gradient Accumulation (Multivariate Chain Rule)**:
           - Local gradient contributions are added to antecedent parents using
             `+=` (`parent.grad = parent.grad + d_parent`).
           - Accumulation (rather than overwriting) is mathematically mandatory
             when a variable is reused across multiple paths (diamond graphs).

        Args:
            grad_tensor: Seed gradient for this node. If None, defaults to
                Tensor(1.0).

        Raises:
            RuntimeError: If called on a non-scalar tensor without an explicit
                `grad_tensor`.
            ValueError: If an operation node has no matching primitive rule in
                `SCALAR_PRIMITIVES`.

        Examples:
            ```python
            import utorch as ut

            # 1. First-order derivative of a polynomial: y = x^2 + 3x
            x = ut.Tensor(4.0, name="x")
            y = (x ** 2) + 3.0 * x  # dy/dx = 2x + 3 = 11.0
            y.backward()
            print(x.grad.value)     # 11.0

            # 2. Second-order derivative (double backward): d2y/dx2 = 2.0
            first_grad = x.grad
            x.zero_grad()
            first_grad.backward()
            print(x.grad.value)     # 2.0
            ```
        """
        # 1. Seed root gradient as a Tensor node
        if grad_tensor is None:
            self.grad = Tensor(1.0)  # type: ignore[assignment]
        else:
            self.grad = grad_tensor  # type: ignore[assignment]

        # 2. Get reverse topological evaluation order (root first, down to leaves)
        ordered_nodes = topological_sort(self)

        # 3. Propagate gradients backwards
        for node in ordered_nodes:
            # Leaf nodes have no parents to propagate into
            if not node.parents:
                continue

            op_key = getattr(node, "op", None) or _get_function_name(node.fun)
            if op_key not in SCALAR_PRIMITIVES:
                raise ValueError(f"No derivative registered for op: '{op_key}'")

            rule = SCALAR_PRIMITIVES[op_key]

            # Binary operations (2 parents)
            if len(node.parents) == 2:
                parent_left, parent_right = node.parents

                # Support both a list [d_left, d_right]
                # or a function returning (d_left, d_right)
                if isinstance(rule, list):
                    d_left = rule[0](node.grad, parent_left, parent_right, node.args)
                    d_right = rule[1](node.grad, parent_left, parent_right, node.args)
                else:
                    d_left, d_right = rule(
                        node.grad, parent_left, parent_right, node.args
                    )

                # Accumulate gradients
                # (+= records computation graph if grad is a Tensor)
                parent_left.grad = (
                    d_left if parent_left.grad is None else parent_left.grad + d_left
                )
                parent_right.grad = (
                    d_right
                    if parent_right.grad is None
                    else parent_right.grad + d_right
                )

            # Unary operations (1 parent)
            elif len(node.parents) == 1:
                parent = node.parents[0]
                rule_fn = rule[0] if isinstance(rule, list) else rule
                d_parent = rule_fn(node.grad, parent, node.args)

                parent.grad = (
                    d_parent if parent.grad is None else parent.grad + d_parent
                )
