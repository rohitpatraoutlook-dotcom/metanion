"""Expression node for Metanion."""

from dataclasses import dataclass
from typing import Optional, Tuple, Any


@dataclass
class ExpressionNode:
    """Expression node representing an operation."""
    op: Any
    left: Optional[int] = None
    right: Optional[int] = None
    
    @property
    def arity(self) -> int:
        """Get arity of the operation."""
        from .op_enum import get_op_arity
        return get_op_arity(self.op)
    
    def get_children(self) -> Tuple[int, ...]:
        """Get child handles."""
        children = []
        if self.left is not None:
            children.append(self.left)
        if self.right is not None:
            children.append(self.right)
        return tuple(children)
    
    def __repr__(self) -> str:
        from .op_enum import get_op_name
        op_name = get_op_name(self.op)
        if self.left is None and self.right is None:
            return f"Node({op_name})"
        elif self.right is None:
            return f"Node({op_name}, {self.left})"
        else:
            return f"Node({op_name}, {self.left}, {self.right})"


class ExpressionNodeFactory:
    """Factory for creating expression nodes."""
    
    @staticmethod
    def create(op, *children: int) -> ExpressionNode:
        """Create an expression node."""
        if len(children) == 0:
            return ExpressionNode(op)
        elif len(children) == 1:
            return ExpressionNode(op, children[0])
        elif len(children) == 2:
            return ExpressionNode(op, children[0], children[1])
        else:
            raise ValueError(f"Too many children: {len(children)}")
    
    @staticmethod
    def create_unary(op, child: int) -> ExpressionNode:
        """Create a unary node."""
        return ExpressionNode(op, child)
    
    @staticmethod
    def create_binary(op, left: int, right: int) -> ExpressionNode:
        """Create a binary node."""
        return ExpressionNode(op, left, right)
    
    @staticmethod
    def create_constant_zero() -> ExpressionNode:
        """Create a zero constant node."""
        from .op_enum import OpID
        return ExpressionNode(OpID.CONST_ZERO)
    
    @staticmethod
    def create_constant_one() -> ExpressionNode:
        """Create a one constant node."""
        from .op_enum import OpID
        return ExpressionNode(OpID.CONST_ONE)
    
    @staticmethod
    def create_identity() -> ExpressionNode:
        """Create an identity node."""
        from .op_enum import OpID
        return ExpressionNode(OpID.IDENTITY)
