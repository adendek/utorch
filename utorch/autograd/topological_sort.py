"""Graph traversal and topological sorting for the autograd engine."""

from __future__ import annotations

from typing import Any


def topological_sort(root: Any) -> list[Any]:
    """Return nodes in reverse topological order (root first, down to leaves).

    Uses DFS with a visited set so shared parent nodes (diamond dependencies)
    are processed only after all their dependent children have been evaluated.

    Args:
        root: The output node from which to trace backwards.

    Returns:
        List of nodes starting with root and ending at leaf nodes.
    """
    ordered: list[Any] = []
    visited: set[int] = set()

    def dfs(node: Any) -> None:
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        for parent in node.parents:
            dfs(parent)
        ordered.append(node)

    dfs(root)
    # Reverse so root is at index 0 and leaves are at the end
    return list(reversed(ordered))
