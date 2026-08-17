"""
Tree printing utilities for the Metanion engine.
Pretty prints expression trees in various formats.
"""

from typing import Optional, List, Dict, Any, Set, Tuple
import textwrap

from ..symbolic import OpID, get_op_name, get_op_arity, lookup, intern
from ..symbolic import get_depth, count_nodes_in_subtree


class TreePrinter:
    """
    Pretty printer for expression trees.
    Supports multiple output formats.
    """
    
    def __init__(self, max_depth: int = 10, max_nodes: int = 50):
        """
        Initialize the tree printer.
        
        Args:
            max_depth: Maximum depth to print.
            max_nodes: Maximum nodes to print.
        """
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self._node_count = 0
    
    def print_tree(
        self,
        handle: int,
        format: str = "text",
        var_name: str = "x",
        indent: int = 2
    ) -> str:
        """
        Print an expression tree in the specified format.
        
        Args:
            handle: The expression handle.
            format: Output format (text, lisp, latex, json, dot).
            var_name: Name of the input variable.
            indent: Indentation size for text format.
            
        Returns:
            String representation of the tree.
        """
        self._node_count = 0
        
        if format == "text":
            return self._to_text(handle, var_name, indent)
        elif format == "lisp":
            return self._to_lisp(handle, var_name)
        elif format == "latex":
            return self._to_latex(handle, var_name)
        elif format == "json":
            return self._to_json(handle, var_name)
        elif format == "dot":
            return self._to_dot(handle, var_name)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _to_text(self, handle: int, var_name: str, indent: int) -> str:
        """Convert to text format."""
        lines = []
        self._text_recurse(handle, var_name, 0, lines, indent)
        return "\n".join(lines)
    
    def _text_recurse(
        self,
        handle: int,
        var_name: str,
        depth: int,
        lines: List[str],
        indent: int
    ) -> None:
        """Recursively build text representation."""
        self._node_count += 1
        
        if self._node_count > self.max_nodes:
            lines.append("  " * depth + "... (truncated)")
            return
        
        if depth > self.max_depth:
            lines.append("  " * depth + "... (max depth)")
            return
        
        node = lookup(handle)
        if node is None:
            lines.append("  " * depth + "None")
            return
        
        op_name = get_op_name(node.op)
        arity = get_op_arity(node.op)
        
        # Base case: constant or variable
        if node.op == OpID.IDENTITY:
            lines.append("  " * depth + f"{var_name}")
            return
        elif node.op == OpID.CONST_ZERO:
            lines.append("  " * depth + "0")
            return
        elif node.op == OpID.CONST_ONE:
            lines.append("  " * depth + "1")
            return
        elif arity == 0:
            lines.append("  " * depth + op_name)
            return
        
        # Get children
        children = node.get_children()
        
        # Show node
        if arity == 1:
            lines.append("  " * depth + f"({op_name})")
            self._text_recurse(children[0], var_name, depth + 1, lines, indent)
        elif arity == 2:
            # Infix operators
            if op_name in ['+', '-', '*', '/', '^', '>', '<', '==']:
                lines.append("  " * depth + f"({op_name})")
                self._text_recurse(children[0], var_name, depth + 1, lines, indent)
                self._text_recurse(children[1], var_name, depth + 1, lines, indent)
            else:
                lines.append("  " * depth + f"({op_name})")
                self._text_recurse(children[0], var_name, depth + 1, lines, indent)
                self._text_recurse(children[1], var_name, depth + 1, lines, indent)
        else:
            lines.append("  " * depth + f"({op_name} arity={arity})")
            for child in children:
                if child is not None:
                    self._text_recurse(child, var_name, depth + 1, lines, indent)
    
    def _to_lisp(self, handle: int, var_name: str) -> str:
        """Convert to Lisp-style S-expression."""
        return self._lisp_recurse(handle, var_name)
    
    def _lisp_recurse(self, handle: int, var_name: str) -> str:
        """Recursively build Lisp representation."""
        self._node_count += 1
        
        if self._node_count > self.max_nodes:
            return "... (truncated)"
        
        if self._node_count > self.max_depth * 2:
            return "... (max depth)"
        
        node = lookup(handle)
        if node is None:
            return "nil"
        
        op_name = get_op_name(node.op)
        
        # Base case
        if node.op == OpID.IDENTITY:
            return var_name
        elif node.op == OpID.CONST_ZERO:
            return "0"
        elif node.op == OpID.CONST_ONE:
            return "1"
        elif node.arity == 0:
            return op_name
        
        # Get children
        children = node.get_children()
        child_strs = [self._lisp_recurse(child, var_name) for child in children if child is not None]
        
        return f"({op_name} {' '.join(child_strs)})"
    
    def _to_latex(self, handle: int, var_name: str) -> str:
        """Convert to LaTeX mathematical notation."""
        return self._latex_recurse(handle, var_name)
    
    def _latex_recurse(self, handle: int, var_name: str) -> str:
        """Recursively build LaTeX representation."""
        self._node_count += 1
        
        if self._node_count > self.max_nodes:
            return "\\dots"
        
        node = lookup(handle)
        if node is None:
            return "\\text{None}"
        
        op_name = get_op_name(node.op)
        arity = get_op_arity(node.op)
        
        # Base case
        if node.op == OpID.IDENTITY:
            return var_name
        elif node.op == OpID.CONST_ZERO:
            return "0"
        elif node.op == OpID.CONST_ONE:
            return "1"
        elif arity == 0:
            return f"\\text{{{op_name}}}"
        
        children = node.get_children()
        
        if arity == 1:
            child_str = self._latex_recurse(children[0], var_name)
            return f"\\{op_name}({child_str})"
        elif arity == 2:
            left = self._latex_recurse(children[0], var_name)
            right = self._latex_recurse(children[1], var_name)
            
            if op_name == '+':
                return f"{left} + {right}"
            elif op_name == '-':
                return f"{left} - {right}"
            elif op_name == '*':
                return f"{left} \\cdot {right}"
            elif op_name == '/':
                return f"\\frac{{{left}}}{{{right}}}"
            elif op_name == '^':
                return f"{{{left}}}^{{{right}}}"
            else:
                return f"\\text{{{op_name}}}({left}, {right})"
        
        return f"\\text{{{op_name}}}(\\dots)"
    
    def _to_json(self, handle: int, var_name: str) -> str:
        """Convert to JSON representation."""
        import json
        return json.dumps(self._json_recurse(handle, var_name), indent=2)
    
    def _json_recurse(self, handle: int, var_name: str) -> Dict[str, Any]:
        """Recursively build JSON representation."""
        self._node_count += 1
        
        if self._node_count > self.max_nodes:
            return {"type": "truncated"}
        
        node = lookup(handle)
        if node is None:
            return {"type": "None"}
        
        result = {
            "op": get_op_name(node.op),
            "op_id": int(node.op),
            "arity": get_op_arity(node.op)
        }
        
        if node.op == OpID.IDENTITY:
            result["type"] = "variable"
            result["name"] = var_name
        elif node.op in (OpID.CONST_ZERO, OpID.CONST_ONE):
            result["type"] = "constant"
            result["value"] = 0 if node.op == OpID.CONST_ZERO else 1
        elif node.arity == 0:
            result["type"] = "constant"
            result["value"] = get_op_name(node.op)
        else:
            result["type"] = "operation"
            children = node.get_children()
            result["children"] = [
                self._json_recurse(child, var_name) if child is not None else None
                for child in children
            ]
        
        return result
    
    def _to_dot(self, handle: int, var_name: str) -> str:
        """Convert to Graphviz DOT format."""
        lines = ['digraph ExpressionTree {', '  node [shape=box];']
        self._dot_recurse(handle, var_name, lines)
        lines.append('}')
        return "\n".join(lines)
    
    def _dot_recurse(
        self,
        handle: int,
        var_name: str,
        lines: List[str],
        parent: Optional[int] = None
    ) -> int:
        """Recursively build DOT representation."""
        self._node_count += 1
        
        if self._node_count > self.max_nodes:
            return handle
        
        node = lookup(handle)
        if node is None:
            return handle
        
        # Create node label
        if node.op == OpID.IDENTITY:
            label = var_name
        elif node.op == OpID.CONST_ZERO:
            label = "0"
        elif node.op == OpID.CONST_ONE:
            label = "1"
        else:
            label = get_op_name(node.op)
        
        node_id = f"n{self._node_count}"
        lines.append(f'  {node_id} [label="{label}"];')
        
        # Connect to parent
        if parent is not None:
            lines.append(f'  {parent} -> {node_id};')
        
        # Recurse on children
        for child in node.get_children():
            if child is not None:
                self._dot_recurse(child, var_name, lines, node_id)
        
        return node_id
    
    def print_expression(self, handle: int, var_name: str = "x") -> str:
        """
        Print a human-readable expression string.
        
        Args:
            handle: The expression handle.
            var_name: Name of the input variable.
            
        Returns:
            Expression string.
        """
        return self._to_lisp(handle, var_name)