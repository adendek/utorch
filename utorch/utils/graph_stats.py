from __future__ import annotations

from dataclasses import dataclass

from utorch.autograd import Tensor


@dataclass(frozen=True)
class GraphStats:
    """Summary statistics for a computational graph.

    Attributes:
        leaves: Number of input/leaf nodes (parameters, inputs, constants).
        interim: Number of operation/intermediate nodes.
        total: Total unique nodes in the graph (leaves + interim).
        edges: Total dependency connections (edges) between nodes.
        ops: Breakdown of operations by frequency (e.g. {'add': 2, 'mul': 1}).
    """

    leaves: int
    interim: int
    total: int
    edges: int
    ops: dict[str, int]

    def __repr__(self) -> str:
        """Return a string representation of the Graph stat for inspection."""
        ops_str = ", ".join(f"{op}: {count}" for op, count in self.ops.items())
        return (
            f"GraphStats(total={self.total}, leaves={self.leaves}, "
            f"interim={self.interim}, edges={self.edges}, ops={{{ops_str}}})"
        )


def count_nodes(root: Tensor) -> GraphStats:
    """Traverse the computational DAG and count leaf and interim nodes.

    Args:
        root: The output node from which to trace the graph backwards.

    Returns:
        GraphStats containing leaf count, interim count, edge count, and op breakdown.

    Example:
        >>> x = Tensor(2.0, name="x")
        >>> y = Tensor(3.0, name="y")
        >>> z = x * y + 1.0
        >>> stats = count_nodes(z)
        >>> stats.leaves
        3  # x, y, and the constant 1.0
        >>> stats.interim
        2  # the '*' node and the '+' node
    """
    seen: set[int] = set()
    leaf_count = 0
    interim_count = 0
    edge_count = 0
    ops: dict[str, int] = {}

    def traverse(node: Tensor) -> None:
        nonlocal leaf_count, interim_count, edge_count

        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)

        if node.is_leaf:
            leaf_count += 1
        else:
            interim_count += 1
            op_name = getattr(node, "op", None) or "func"
            ops[op_name] = ops.get(op_name, 0) + 1

        for parent in node.parents:
            edge_count += 1
            traverse(parent)

    traverse(root)

    return GraphStats(
        leaves=leaf_count,
        interim=interim_count,
        total=leaf_count + interim_count,
        edges=edge_count,
        ops=ops,
    )
