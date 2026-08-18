"""Expression node for Metanion."""

from dataclasses import dataclass
from typing import Optional, Tuple, Any
from .op_enum import OpID, get_op_arity, get_op_name


@dataclass
class ExpressionNode:
    op: OpID
    left: Optional[int] = None
    right: Optional[int] = None
    value: Optional[float] = None   # for CONST
    index: Optional[int] = None     # for VAR

    def __post_init__(self):
        if self.op == OpID.CONST:
            if self.value is None:
                raise ValueError("CONST must have a value")
            if self.left is not None or self.right is not None or self.index is not None:
                raise ValueError("CONST cannot have children or index")
        elif self.op == OpID.VAR:
            if self.index is None:
                raise ValueError("VAR must have an index")
            if self.left is not None or self.right is not None or self.value is not None:
                raise ValueError("VAR cannot have children or value")
        else:
            arity = get_op_arity(self.op)
            if arity == 0:
                if self.left is not None or self.right is not None or self.index is not None:
                    raise ValueError(f"Operation {get_op_name(self.op)} cannot have children or index")
            elif arity == 1:
                if self.left is None:
                    raise ValueError(f"Unary operation {get_op_name(self.op)} requires left child")
                if self.right is not None:
                    raise ValueError(f"Unary operation {get_op_name(self.op)} cannot have right child")
            elif arity == 2:
                if self.left is None or self.right is None:
                    raise ValueError(f"Binary operation {get_op_name(self.op)} requires both children")
            else:
                raise ValueError(f"Unsupported arity {arity} for {get_op_name(self.op)}")

    def get_children(self) -> Tuple[int, ...]:
        if self.op in [OpID.CONST, OpID.VAR]:
            return ()
        children = []
        if self.left is not None:
            children.append(self.left)
        if self.right is not None:
            children.append(self.right)
        return tuple(children)

    def __repr__(self) -> str:
        if self.op == OpID.CONST:
            return f"Node(CONST, value={self.value})"
        if self.op == OpID.VAR:
            return f"Node(VAR, index={self.index})"
        return f"Node({get_op_name(self.op)}, {self.left}, {self.right})"


class ExpressionNodeFactory:
    @staticmethod
    def create(op: OpID, *children: int) -> ExpressionNode:
        if len(children) == 0:
            return ExpressionNode(op)
        elif len(children) == 1:
            return ExpressionNode(op, children[0])
        elif len(children) == 2:
            return ExpressionNode(op, children[0], children[1])
        else:
            raise ValueError(f"Too many children: {len(children)}")
    
    @staticmethod
    def create_const(value: float) -> ExpressionNode:
        return ExpressionNode(OpID.CONST, value=value)
    
    @staticmethod
    def create_var(index: int) -> ExpressionNode:
        return ExpressionNode(OpID.VAR, index=index)
    
    @staticmethod
    def create_identity() -> ExpressionNode:
        return ExpressionNode(OpID.IDENTITY)
    
    @staticmethod
    def create_zero() -> ExpressionNode:
        return ExpressionNode(OpID.CONST_ZERO)
    
    @staticmethod
    def create_one() -> ExpressionNode:
        return ExpressionNode(OpID.CONST_ONE)
