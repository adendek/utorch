"""Visualization utilities for computational graphs."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from graphviz import Digraph

from utorch.autograd.tensor import Tensor


def _get_clean_op_name(node: Tensor) -> str:
    """Extract a clean, human-readable operation name from a node.

    Converts:
        <function Tensor.__add__.<locals>.<lambda> at 0x10851f2e0> -> "add"
        <function Tensor.sin at 0x...>                             -> "sin"
        <ufunc 'add'>                                              -> "add"
    """
    # 1. Use explicit op attribute if present
    if hasattr(node, "op") and node.op:
        return str(node.op)

    fun = getattr(node, "fun", None)
    if fun is None:
        return "op"

    # 2. If it's a standard named function or ufunc
    if hasattr(fun, "__name__") and fun.__name__ != "<lambda>":
        return str(fun.__name__)

    # 3. Clean up lambdas like Tensor.__add__.<locals>.<lambda>
    fun_str = str(fun)
    match = re.search(r"__([a-zA-Z0-9_]+)__", fun_str)
    if match:
        raw_op = match.group(1)
        # Strip common dunder prefixes (e.g., radd -> add, truediv -> div)
        op_mapping = {
            "radd": "add",
            "rmul": "mul",
            "rsub": "sub",
            "rtruediv": "div",
            "truediv": "div",
            "matmul": "matmul",
            "pow": "pow",
            "neg": "neg",
        }
        return op_mapping.get(raw_op, raw_op)

    # 4. Fallback for custom objects / ufuncs
    return fun_str.strip("<>").split()[0].split(".")[-1]


def draw_graph(
    root: Tensor,
    format: str = "png",
    rankdir: str = "LR",
) -> Any:
    """Render a computational graph with a tree-inspired aesthetic.

    - Leaf nodes (inputs, constants, parameters) are rendered in green.
    - Interim nodes (operations, intermediate results) are rendered in brown.

    Args:
        root: The output node (root of the evaluation) to trace backwards.
        format: Output rendering format ('svg', 'png', 'pdf').
        rankdir: Graph layout direction ('LR' = left-to-right, 'TB' = top-to-bottom).

    Returns:
        graphviz.Digraph object ready for notebook rendering or file saving.
    """

    # Graph-level attributes
    dot = Digraph(
        name="computation_graph",
        format=format,
        graph_attr={
            "rankdir": rankdir,
            "splines": "ortho",
            "nodesep": "0.4",
            "ranksep": "0.6",
            "bgcolor": "transparent",
        },
        node_attr={
            "style": "filled,rounded",
            "shape": "box",
            "fontname": "Helvetica, Arial, sans-serif",
            "fontsize": "11",
            "penwidth": "1.5",
        },
        edge_attr={
            "color": "#5D4037",
            "arrowsize": "0.8",
            "penwidth": "1.2",
        },
    )

    seen: set[int] = set()

    def format_value(val: Any) -> str:
        """Format node values cleanly for diagram labels."""
        if isinstance(val, np.ndarray):
            if val.size == 1:
                return f"{val.item():.4g}"
            return f"array(shape={val.shape})"
        if isinstance(val, int | float):
            return f"{val:.4g}"
        return str(val)

    def build(node: Tensor) -> None:
        node_id = str(id(node))
        if id(node) in seen:
            return
        seen.add(id(node))

        val_str = format_value(node.value)

        if node.is_leaf:
            # --- LEAF NODE (Green) ---
            title = f"{node.name}\n" if node.name else ""
            label = f"{title}data: {val_str}"
            dot.node(
                node_id,
                label=label,
                fillcolor="#C8E6C9",  # Soft leaf green
                color="#2E7D32",  # Forest green border
                fontcolor="#1B5E20",  # Dark green text
            )
        else:
            # --- INTERIM NODE (Woody Brown) ---
            op_name = _get_clean_op_name(node)
            op_str = f"fun: {op_name}" if node.fun else "op: func"
            name_str = f"{node.name} | " if node.name else ""
            label = f"{name_str}{op_str}\nval: {val_str}"
            dot.node(
                node_id,
                label=label,
                fillcolor="#D7CCC8",  # Light wood brown
                color="#5D4037",  # Bark brown border
                fontcolor="#3E2723",  # Deep espresso text
            )

        # Draw directed edges from parents -> current node
        for parent in node.parents:
            parent_id = str(id(parent))
            dot.edge(parent_id, node_id)
            build(parent)

    build(root)
    return dot
