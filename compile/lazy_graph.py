"""
Lazy operation graph for the Metanion engine.
Implements deferred execution of tensor operations.
"""

from typing import Optional, Union, List, Tuple, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import weakref

from ..core.tensor import Tensor
from ..core.tensor_shape import Shape, ShapeTracker
from ..core.dtype_system import DType, promote_dtypes
from ..symbolic import OpID, intern, get_op_arity, get_op_name
from ..exceptions import ExpressionError, ShapeMismatchError


class LazyOpType(Enum):
    """Types of lazy operations."""
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    NEG = "neg"
    EXP = "exp"
    LOG = "log"
    SIN = "sin"
    COS = "cos"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    RELU = "relu"
    SQUARE = "square"
    SQRT = "sqrt"
    IDENTITY = "identity"
    REDUCE_SUM = "reduce_sum"
    REDUCE_MEAN = "reduce_mean"
    RESHAPE = "reshape"
    TRANSPOSE = "transpose"
    BROADCAST = "broadcast"
    
    @classmethod
    def from_string(cls, name: str) -> 'LazyOpType':
        """Get LazyOpType from string name."""
        mapping = {
            'add': cls.ADD,
            'sub': cls.SUB,
            'mul': cls.MUL,
            'div': cls.DIV,
            'pow': cls.POW,
            'neg': cls.NEG,
            'exp': cls.EXP,
            'log': cls.LOG,
            'sin': cls.SIN,
            'cos': cls.COS,
            'tanh': cls.TANH,
            'sigmoid': cls.SIGMOID,
            'relu': cls.RELU,
            'square': cls.SQUARE,
            'sqrt': cls.SQRT,
            'identity': cls.IDENTITY,
            'sum': cls.REDUCE_SUM,
            'mean': cls.REDUCE_MEAN,
            'reshape': cls.RESHAPE,
            'transpose': cls.TRANSPOSE,
            'broadcast': cls.BROADCAST,
        }
        if name not in mapping:
            raise ValueError(f"Unknown operation: {name}")
        return mapping[name]
    
    def to_op_id(self) -> OpID:
        """Convert to symbolic OpID."""
        mapping = {
            self.ADD: OpID.ADD,
            self.SUB: OpID.SUB,
            self.MUL: OpID.MUL,
            self.DIV: OpID.DIV,
            self.POW: OpID.POWER,
            self.NEG: OpID.NEG,
            self.EXP: OpID.EXP,
            self.LOG: OpID.LOG,
            self.SIN: OpID.SIN,
            self.COS: OpID.COS,
            self.TANH: OpID.TANH,
            self.SIGMOID: OpID.SIGMOID,
            self.RELU: OpID.RELU,
            self.SQUARE: OpID.SQUARE,
            self.SQRT: OpID.SQRT,
            self.IDENTITY: OpID.IDENTITY,
        }
        if self not in mapping:
            raise ValueError(f"No OpID mapping for {self}")
        return mapping[self]


@dataclass
class LazyOp:
    """
    Lazy operation node in the computation graph.
    Represents a deferred operation that will be executed later.
    """
    
    op_type: LazyOpType
    left: Union[Tensor, 'LazyOp', None] = None
    right: Union[Tensor, 'LazyOp', None] = None
    shape: Optional[Shape] = None
    dtype: Optional[DType] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize computed fields."""
        self._handle: Optional[int] = None
        self._compiled: bool = False
        self._children: Set[int] = set()
        
        # Infer shape if not provided
        if self.shape is None:
            self._infer_shape()
        
        # Infer dtype if not provided
        if self.dtype is None:
            self._infer_dtype()
    
    def _infer_shape(self) -> None:
        """Infer output shape from inputs."""
        if self.op_type in (LazyOpType.IDENTITY, LazyOpType.NEG, 
                           LazyOpType.EXP, LazyOpType.LOG,
                           LazyOpType.SIN, LazyOpType.COS,
                           LazyOpType.TANH, LazyOpType.SIGMOID,
                           LazyOpType.RELU, LazyOpType.SQUARE,
                           LazyOpType.SQRT):
            # Unary ops preserve shape
            left_shape = self._get_shape(self.left)
            self.shape = left_shape
            return
        
        if self.op_type in (LazyOpType.ADD, LazyOpType.SUB,
                           LazyOpType.MUL, LazyOpType.DIV,
                           LazyOpType.POW):
            # Binary ops broadcast
            left_shape = self._get_shape(self.left)
            right_shape = self._get_shape(self.right)
            self.shape = ShapeTracker.broadcast_shape(left_shape, right_shape)
            return
        
        if self.op_type == LazyOpType.RESHAPE:
            # Reshape op
            new_shape = self.kwargs.get('shape')
            if new_shape is None:
                raise ExpressionError("Reshape requires 'shape' in kwargs")
            left_shape = self._get_shape(self.left)
            self.shape = ShapeTracker.reshape(left_shape, new_shape)
            return
        
        if self.op_type == LazyOpType.TRANSPOSE:
            # Transpose op
            left_shape = self._get_shape(self.left)
            axes = self.kwargs.get('axes')
            if axes is None:
                # Default: reverse dimensions
                axes = tuple(reversed(range(len(left_shape))))
            # TODO: Implement transpose shape computation
            self.shape = left_shape
            return
        
        if self.op_type in (LazyOpType.REDUCE_SUM, LazyOpType.REDUCE_MEAN):
            # Reduction ops
            left_shape = self._get_shape(self.left)
            axis = self.kwargs.get('axis')
            keepdims = self.kwargs.get('keepdims', False)
            
            if axis is None:
                # Reduce all dimensions
                if keepdims:
                    self.shape = (1,) * len(left_shape)
                else:
                    self.shape = ()
            else:
                # Reduce specific axis
                if isinstance(axis, int):
                    axis = (axis,)
                result = list(left_shape)
                for ax in axis:
                    if keepdims:
                        result[ax] = 1
                    else:
                        result[ax] = -1  # Will be removed
                if not keepdims:
                    result = [d for d in result if d != -1]
                self.shape = tuple(result)
            return
        
        if self.op_type == LazyOpType.BROADCAST:
            # Broadcast to a target shape
            target_shape = self.kwargs.get('target_shape')
            if target_shape is None:
                raise ExpressionError("Broadcast requires 'target_shape' in kwargs")
            left_shape = self._get_shape(self.left)
            self.shape = ShapeTracker.broadcast_shape(left_shape, target_shape)
            return
        
        # Default: unknown shape
        self.shape = ()
    
    def _infer_dtype(self) -> None:
        """Infer output dtype from inputs."""
        if self.left is not None:
            left_dtype = self._get_dtype(self.left)
            if self.right is not None:
                right_dtype = self._get_dtype(self.right)
                self.dtype = promote_dtypes(left_dtype, right_dtype)
            else:
                self.dtype = left_dtype
        else:
            self.dtype = DType.FLOAT64
    
    @staticmethod
    def _get_shape(obj: Union[Tensor, 'LazyOp', None]) -> Shape:
        """Get shape from tensor or lazy op."""
        if obj is None:
            return ()
        if isinstance(obj, Tensor):
            return obj.shape
        if isinstance(obj, LazyOp):
            if obj.shape is None:
                raise ExpressionError(f"LazyOp {obj.op_type} has no shape")
            return obj.shape
        raise TypeError(f"Unsupported type: {type(obj)}")
    
    @staticmethod
    def _get_dtype(obj: Union[Tensor, 'LazyOp', None]) -> DType:
        """Get dtype from tensor or lazy op."""
        if obj is None:
            return DType.FLOAT64
        if isinstance(obj, Tensor):
            return obj.dtype
        if isinstance(obj, LazyOp):
            if obj.dtype is None:
                return DType.FLOAT64
            return obj.dtype
        raise TypeError(f"Unsupported type: {type(obj)}")
    
    def build_expression(self) -> int:
        """
        Build the symbolic expression from this lazy op.
        Returns a handle to the expression in the pool.
        """
        if self._handle is not None:
            return self._handle
        
        # Build left and right expressions
        left_handle = None
        right_handle = None
        
        if self.left is not None:
            if isinstance(self.left, LazyOp):
                left_handle = self.left.build_expression()
            elif isinstance(self.left, Tensor):
                # Tensor as leaf - use identity
                left_handle = intern(OpID.IDENTITY)
            else:
                raise ExpressionError(f"Unsupported left type: {type(self.left)}")
        
        if self.right is not None:
            if isinstance(self.right, LazyOp):
                right_handle = self.right.build_expression()
            elif isinstance(self.right, Tensor):
                right_handle = intern(OpID.IDENTITY)
            else:
                raise ExpressionError(f"Unsupported right type: {type(self.right)}")
        
        # Create the operation node
        op_id = self.op_type.to_op_id()
        
        # Get arity and intern
        arity = get_op_arity(op_id)
        if arity == 1:
            self._handle = intern(op_id, left_handle)
        elif arity == 2:
            self._handle = intern(op_id, left_handle, right_handle)
        else:
            raise ExpressionError(f"Unsupported arity: {arity}")
        
        return self._handle
    
    def evaluate(self) -> Tensor:
        """
        Evaluate the lazy operation and return a Tensor.
        This performs the actual computation.
        """
        # Build the expression first
        handle = self.build_expression()
        
        # TODO: Actual evaluation with the JIT compiler
        # For now, return a placeholder tensor
        from ..core.tensor import Tensor
        result = Tensor.zeros(self.shape, self.dtype)
        return result
    
    def __repr__(self) -> str:
        """String representation."""
        op_name = self.op_type.value
        if self.left is not None:
            if self.right is not None:
                return f"LazyOp({op_name}, {self.left}, {self.right}, shape={self.shape})"
            return f"LazyOp({op_name}, {self.left}, shape={self.shape})"
        return f"LazyOp({op_name}, shape={self.shape})"


class LazyGraph:
    """
    Lazy computation graph for building and optimizing expressions.
    """
    
    def __init__(self):
        """Initialize an empty graph."""
        self._ops: List[LazyOp] = []
        self._variables: Dict[str, Tensor] = {}
        self._output: Optional[LazyOp] = None
        self._compiled: bool = False
    
    def add_variable(self, name: str, tensor: Tensor) -> None:
        """Add a variable to the graph."""
        self._variables[name] = tensor
    
    def add_op(self, op: LazyOp) -> None:
        """Add an operation to the graph."""
        self._ops.append(op)
        self._output = op
        self._compiled = False
    
    def get_output(self) -> Optional[LazyOp]:
        """Get the output operation."""
        return self._output
    
    def compile(self) -> int:
        """
        Compile the graph into a single expression.
        Returns a handle to the compiled expression.
        """
        if self._output is None:
            raise ExpressionError("No output operation in graph")
        
        self._compiled = True
        return self._output.build_expression()
    
    def simplify(self) -> None:
        """Simplify the graph by removing redundant operations."""
        # TODO: Implement graph simplification
        pass
    
    def __repr__(self) -> str:
        """String representation."""
        return f"LazyGraph(ops={len(self._ops)}, compiled={self._compiled})"