"""
Expression node representation for the Metanion engine.
Defines the structure of AST nodes and utility functions for manipulation.
"""

from typing import Optional, Tuple, List, Union, Any
from dataclasses import dataclass, field
from functools import total_ordering

from .op_enum import OpID, get_op_arity, get_op_name, is_binary_op, is_unary_op
from ..exceptions import ExpressionError, TypeSignatureError


@total_ordering
@dataclass(frozen=True)
class ExpressionNode:
    """
    Immutable expression node representing a single operation application.
    This is the core data structure stored in the hash-consing pool.
    """
    
    op: OpID
    left: Optional[int] = None      # Handle of left child (or only child for unary)
    right: Optional[int] = None     # Handle of right child (None for unary)
    
    def __post_init__(self):
        """Validate the node structure."""
        arity = get_op_arity(self.op)
        
        if arity == 1:
            if self.left is None:
                raise ExpressionError(f"Unary operation {get_op_name(self.op)} requires left child")
            if self.right is not None:
                raise ExpressionError(f"Unary operation {get_op_name(self.op)} cannot have right child")
        
        elif arity == 2:
            if self.left is None or self.right is None:
                raise ExpressionError(f"Binary operation {get_op_name(self.op)} requires both children")
        
        elif arity == 3:
            if self.left is None or self.right is None:
                raise ExpressionError(f"Ternary operation {get_op_name(self.op)} requires both children")
            # Note: Some ternary ops might have 3 children, but we use left/right as first two
            # and store third in a separate field if needed. For simplicity, we'll handle
            # ternary via a special field or separate class if needed.
        
        elif arity == 0:
            if self.left is not None or self.right is not None:
                raise ExpressionError(f"Nullary operation {get_op_name(self.op)} cannot have children")
        
        else:
            raise ExpressionError(f"Unsupported arity {arity} for operation {get_op_name(self.op)}")
    
    @property
    def arity(self) -> int:
        """Get the arity of this node's operation."""
        return get_op_arity(self.op)
    
    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return self.arity == 0
    
    @property
    def is_unary(self) -> bool:
        """Check if this node represents a unary operation."""
        return self.arity == 1
    
    @property
    def is_binary(self) -> bool:
        """Check if this node represents a binary operation."""
        return self.arity == 2
    
    def get_children(self) -> Tuple[int, ...]:
        """Get all child handles as a tuple."""
        children = []
        if self.left is not None:
            children.append(self.left)
        if self.right is not None:
            children.append(self.right)
        return tuple(children)
    
    def get_child_count(self) -> int:
        """Get the number of children."""
        return len(self.get_children())
    
    def replace_child(self, old_handle: int, new_handle: int) -> 'ExpressionNode':
        """
        Replace a child handle with a new one.
        
        Args:
            old_handle: Handle to replace.
            new_handle: New handle.
            
        Returns:
            New ExpressionNode with replaced child.
            
        Raises:
            ExpressionError: If old_handle is not a child.
        """
        if self.left == old_handle:
            return ExpressionNode(self.op, new_handle, self.right)
        elif self.right == old_handle:
            return ExpressionNode(self.op, self.left, new_handle)
        else:
            raise ExpressionError(f"Handle {old_handle} not found in node")
    
    def to_tuple(self) -> Tuple:
        """Convert to a flat tuple for hashing."""
        if self.arity == 0:
            return (self.op,)
        elif self.arity == 1:
            return (self.op, self.left)
        elif self.arity == 2:
            return (self.op, self.left, self.right)
        else:
            return (self.op, self.left, self.right)
    
    def __lt__(self, other: 'ExpressionNode') -> bool:
        """Define ordering for total_ordering."""
        if not isinstance(other, ExpressionNode):
            return NotImplemented
        return self.to_tuple() < other.to_tuple()
    
    def __eq__(self, other: Any) -> bool:
        """Check equality with another node."""
        if not isinstance(other, ExpressionNode):
            return False
        return self.to_tuple() == other.to_tuple()
    
    def __hash__(self) -> int:
        """Compute hash for this node."""
        return hash(self.to_tuple())
    
    def __repr__(self) -> str:
        """String representation."""
        if self.arity == 0:
            return f"Node({get_op_name(self.op)})"
        elif self.arity == 1:
            return f"Node({get_op_name(self.op)}, {self.left})"
        elif self.arity == 2:
            return f"Node({get_op_name(self.op)}, {self.left}, {self.right})"
        else:
            return f"Node({get_op_name(self.op)}, {self.left}, {self.right}, ...)"


class ExpressionNodeFactory:
    """
    Factory for creating expression nodes with validation.
    Provides convenience methods for common operations.
    """
    
    @staticmethod
    def create(op: OpID, *children: int) -> ExpressionNode:
        """
        Create an expression node with the given operation and children.
        
        Args:
            op: Operation ID.
            *children: Child handles (arity must match operation).
            
        Returns:
            ExpressionNode instance.
            
        Raises:
            TypeSignatureError: If arity mismatch.
        """
        arity = get_op_arity(op)
        
        if len(children) != arity:
            raise TypeSignatureError(
                f"Operation {get_op_name(op)} expects {arity} arguments, "
                f"got {len(children)}"
            )
        
        if arity == 0:
            return ExpressionNode(op)
        elif arity == 1:
            return ExpressionNode(op, children[0], None)
        elif arity == 2:
            return ExpressionNode(op, children[0], children[1])
        else:
            # For arity > 2, store first two in left/right and others in a separate field
            # This simplified version only handles arity <= 2
            raise TypeSignatureError(f"Unsupported arity: {arity}")
    
    @staticmethod
    def create_unary(op: OpID, child: int) -> ExpressionNode:
        """Create a unary node."""
        if get_op_arity(op) != 1:
            raise TypeSignatureError(f"Operation {get_op_name(op)} is not unary")
        return ExpressionNode(op, child, None)
    
    @staticmethod
    def create_binary(op: OpID, left: int, right: int) -> ExpressionNode:
        """Create a binary node."""
        if get_op_arity(op) != 2:
            raise TypeSignatureError(f"Operation {get_op_name(op)} is not binary")
        return ExpressionNode(op, left, right)
    
    @staticmethod
    def create_constant_zero() -> ExpressionNode:
        """Create a node representing the constant 0."""
        return ExpressionNode(OpID.CONST_ZERO)
    
    @staticmethod
    def create_constant_one() -> ExpressionNode:
        """Create a node representing the constant 1."""
        return ExpressionNode(OpID.CONST_ONE)
    
    @staticmethod
    def create_identity() -> ExpressionNode:
        """Create a node representing the identity function."""
        return ExpressionNode(OpID.IDENTITY)
    
    @staticmethod
    def create_add(left: int, right: int) -> ExpressionNode:
        """Create an addition node."""
        return ExpressionNode(OpID.ADD, left, right)
    
    @staticmethod
    def create_mul(left: int, right: int) -> ExpressionNode:
        """Create a multiplication node."""
        return ExpressionNode(OpID.MUL, left, right)
    
    @staticmethod
    def create_sub(left: int, right: int) -> ExpressionNode:
        """Create a subtraction node."""
        return ExpressionNode(OpID.SUB, left, right)
    
    @staticmethod
    def create_div(left: int, right: int) -> ExpressionNode:
        """Create a division node."""
        return ExpressionNode(OpID.DIV, left, right)
    
    @staticmethod
    def create_power(base: int, exponent: int) -> ExpressionNode:
        """Create a power node."""
        return ExpressionNode(OpID.POWER, base, exponent)
    
    @staticmethod
    def create_exp(child: int) -> ExpressionNode:
        """Create an exponential node."""
        return ExpressionNode(OpID.EXP, child, None)
    
    @staticmethod
    def create_log(child: int) -> ExpressionNode:
        """Create a logarithm node."""
        return ExpressionNode(OpID.LOG, child, None)
    
    @staticmethod
    def create_sin(child: int) -> ExpressionNode:
        """Create a sine node."""
        return ExpressionNode(OpID.SIN, child, None)
    
    @staticmethod
    def create_cos(child: int) -> ExpressionNode:
        """Create a cosine node."""
        return ExpressionNode(OpID.COS, child, None)
    
    @staticmethod
    def create_tanh(child: int) -> ExpressionNode:
        """Create a hyperbolic tangent node."""
        return ExpressionNode(OpID.TANH, child, None)


# Utility functions for working with expression nodes

def is_constant_expression(node: ExpressionNode) -> bool:
    """Check if an expression node represents a constant."""
    from .op_enum import is_constant_op
    return is_constant_op(node.op)


def is_identity_expression(node: ExpressionNode) -> bool:
    """Check if an expression node represents the identity function."""
    return node.op == OpID.IDENTITY


def get_node_depth(node: ExpressionNode, handle_lookup) -> int:
    """
    Compute the depth of an expression node.
    
    Args:
        node: The expression node.
        handle_lookup: Function to lookup a handle's node.
        
    Returns:
        Depth (number of nodes along the longest path).
    """
    if node.arity == 0:
        return 1
    
    max_child_depth = 0
    for child in node.get_children():
        if child is not None:
            child_node = handle_lookup(child)
            if child_node is not None:
                max_child_depth = max(max_child_depth, get_node_depth(child_node, handle_lookup))
    
    return 1 + max_child_depth


def count_nodes(node: ExpressionNode, handle_lookup) -> int:
    """
    Count the total number of nodes in an expression.
    
    Args:
        node: The expression node.
        handle_lookup: Function to lookup a handle's node.
        
    Returns:
        Total node count.
    """
    total = 1
    for child in node.get_children():
        if child is not None:
            child_node = handle_lookup(child)
            if child_node is not None:
                total += count_nodes(child_node, handle_lookup)
    return total


def get_variable_handles(node: ExpressionNode, handle_lookup) -> List[int]:
    """
    Get all variable handles used in an expression.
    
    Args:
        node: The expression node.
        handle_lookup: Function to lookup a handle's node.
        
    Returns:
        List of variable handles.
    """
    variables = []
    
    if node.op == OpID.IDENTITY:
        # IDENTITY is our variable placeholder
        variables.append(id(node))  # This is a placeholder - actual handle is unknown
    
    for child in node.get_children():
        if child is not None:
            child_node = handle_lookup(child)
            if child_node is not None:
                variables.extend(get_variable_handles(child_node, handle_lookup))
    
    return variables


def node_to_string(node: ExpressionNode, handle_lookup, var_name: str = "x") -> str:
    """
    Convert an expression node to a human-readable string.
    
    Args:
        node: The expression node.
        handle_lookup: Function to lookup a handle's node.
        var_name: Name of the input variable.
        
    Returns:
        String representation.
    """
    from .op_enum import get_op_name
    
    if node.op == OpID.IDENTITY:
        return var_name
    elif node.op == OpID.CONST_ZERO:
        return "0"
    elif node.op == OpID.CONST_ONE:
        return "1"
    
    op_name = get_op_name(node.op)
    
    if node.arity == 0:
        return op_name
    elif node.arity == 1:
        if node.left is not None:
            child_node = handle_lookup(node.left)
            child_str = node_to_string(child_node, handle_lookup, var_name) if child_node else f"<{node.left}>"
            return f"{op_name}({child_str})"
        else:
            return f"{op_name}(?)"
    elif node.arity == 2:
        left_str = "?"
        right_str = "?"
        
        if node.left is not None:
            child_node = handle_lookup(node.left)
            left_str = node_to_string(child_node, handle_lookup, var_name) if child_node else f"<{node.left}>"
        
        if node.right is not None:
            child_node = handle_lookup(node.right)
            right_str = node_to_string(child_node, handle_lookup, var_name) if child_node else f"<{node.right}>"
        
        # For infix operators, use infix notation
        if op_name in ['+', '-', '*', '/', '^', '>', '<', '==', 'and', 'or']:
            return f"({left_str} {op_name} {right_str})"
        else:
            return f"{op_name}({left_str}, {right_str})"
    else:
        return f"{op_name}(...)"