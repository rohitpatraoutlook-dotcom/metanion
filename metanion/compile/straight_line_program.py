"""
Straight-line program (SLP) representation for the Metanion engine.
Converts expression trees to linear sequences of operations for fast evaluation.
"""

from typing import Optional, List, Tuple, Dict, Set, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import math

from ..symbolic import OpID, get_op_arity, get_op_name, get_op_metadata
from ..symbolic import get_pool, lookup, get_depth, count_nodes_in_subtree
from ..exceptions import ExpressionError, CompilationError


class SLPOpType(Enum):
    """Types of operations in a straight-line program."""
    # Arithmetic
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    POW = "pow"
    NEG = "neg"
    
    # Exponential/Log
    EXP = "exp"
    LOG = "log"
    EXP2 = "exp2"
    LOG2 = "log2"
    
    # Trigonometric
    SIN = "sin"
    COS = "cos"
    TAN = "tan"
    TANH = "tanh"
    ASIN = "asin"
    ACOS = "acos"
    ATAN = "atan"
    
    # Activation
    SIGMOID = "sigmoid"
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SWISH = "swish"
    SOFTPLUS = "softplus"
    
    # Special
    SQRT = "sqrt"
    CBRT = "cbrt"
    ABS = "abs"
    INVERSE = "inverse"
    SQUARE = "square"
    CUBE = "cube"
    ERF = "erf"
    
    # Logical
    WHERE = "where"
    GREATER = "greater"
    LESS = "less"
    EQUAL = "equal"
    
    # Constants
    CONST_ZERO = "const_zero"
    CONST_ONE = "const_one"
    CONST = "const"  # General constant (if needed)
    
    # Identity
    IDENTITY = "identity"
    
    # Reduction
    SUM = "sum"
    MEAN = "mean"
    VAR = "var"
    STD = "std"
    MAX = "max"
    MIN = "min"
    
    # Flow control
    DERIVATIVE = "derivative"
    STOP_GRADIENT = "stop_gradient"


@dataclass
class SLPInstruction:
    """
    A single instruction in a straight-line program.
    """
    op: SLPOpType
    arg1: Optional[int] = None      # First argument (register or constant)
    arg2: Optional[int] = None      # Second argument (register or constant)
    dest: Optional[int] = None      # Destination register
    const_val: Optional[float] = None  # Constant value (if op is CONST)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the instruction."""
        if self.op == SLPOpType.CONST:
            if self.const_val is None:
                raise CompilationError("CONST instruction requires const_val")
        
        # Count arguments based on op type
        arity_map = {
            SLPOpType.CONST_ZERO: 0,
            SLPOpType.CONST_ONE: 0,
            SLPOpType.CONST: 0,
            SLPOpType.IDENTITY: 1,
            SLPOpType.NEG: 1,
            SLPOpType.EXP: 1,
            SLPOpType.LOG: 1,
            SLPOpType.SIN: 1,
            SLPOpType.COS: 1,
            SLPOpType.TAN: 1,
            SLPOpType.TANH: 1,
            SLPOpType.ASIN: 1,
            SLPOpType.ACOS: 1,
            SLPOpType.ATAN: 1,
            SLPOpType.ABS: 1,
            SLPOpType.SQRT: 1,
            SLPOpType.CBRT: 1,
            SLPOpType.INVERSE: 1,
            SLPOpType.SQUARE: 1,
            SLPOpType.CUBE: 1,
            SLPOpType.ERF: 1,
            SLPOpType.SIGMOID: 1,
            SLPOpType.RELU: 1,
            SLPOpType.LEAKY_RELU: 1,
            SLPOpType.GELU: 1,
            SLPOpType.SWISH: 1,
            SLPOpType.SOFTPLUS: 1,
            SLPOpType.EXP2: 1,
            SLPOpType.LOG2: 1,
            SLPOpType.STOP_GRADIENT: 1,
            SLPOpType.SUM: 1,
            SLPOpType.MEAN: 1,
            SLPOpType.VAR: 1,
            SLPOpType.STD: 1,
            SLPOpType.MAX: 1,
            SLPOpType.MIN: 1,
            SLPOpType.DERIVATIVE: 1,
            SLPOpType.ADD: 2,
            SLPOpType.SUB: 2,
            SLPOpType.MUL: 2,
            SLPOpType.DIV: 2,
            SLPOpType.POW: 2,
            SLPOpType.WHERE: 3,
            SLPOpType.GREATER: 2,
            SLPOpType.LESS: 2,
            SLPOpType.EQUAL: 2,
        }
        
        expected_arity = arity_map.get(self.op, 0)
        actual_arity = 0
        if self.arg1 is not None:
            actual_arity += 1
        if self.arg2 is not None:
            actual_arity += 1
        
        if actual_arity != expected_arity:
            raise CompilationError(
                f"Instruction {self.op.value} expects {expected_arity} arguments, "
                f"got {actual_arity}"
            )
    
    def __repr__(self) -> str:
        """String representation."""
        if self.op == SLPOpType.CONST:
            return f"SLP({self.op.value}, {self.const_val:.4f}, -> reg{self.dest})"
        elif self.op in (SLPOpType.CONST_ZERO, SLPOpType.CONST_ONE):
            return f"SLP({self.op.value}, -> reg{self.dest})"
        elif self.arg1 is not None and self.arg2 is not None:
            return f"SLP({self.op.value}, reg{self.arg1}, reg{self.arg2}, -> reg{self.dest})"
        elif self.arg1 is not None:
            return f"SLP({self.op.value}, reg{self.arg1}, -> reg{self.dest})"
        else:
            return f"SLP({self.op.value}, -> reg{self.dest})"


class StraightLineProgram:
    """
    Straight-line program representation of an expression tree.
    Converts a symbolic expression into a linear sequence of operations.
    """
    
    def __init__(self, handle: int):
        """
        Initialize a straight-line program from a handle.
        
        Args:
            handle: Handle to the expression in the pool.
        """
        self._handle = handle
        self._instructions: List[SLPInstruction] = []
        self._register_map: Dict[int, int] = {}  # handle -> register number
        self._max_register: int = -1
        self._constants: Dict[int, float] = {}   # handle -> const value
        self._compiled: bool = False
        
        # Build the program
        self._build()
    
    def _build(self) -> None:
        """Build the straight-line program from the expression tree."""
        pool = get_pool()
        
        # Get the expression node
        node = lookup(self._handle)
        if node is None:
            raise HandleNotFoundError(f"Handle {self._handle} not found")
        
        # Collect all handles in postorder
        from ..symbolic import HandleTraversal
        handles = HandleTraversal.postorder_iter(self._handle, lookup)
        
        # Process each handle
        for handle in handles:
            node = lookup(handle)
            if node is None:
                continue
            
            # Check if this is a constant
            op_id = node.op
            op_name = get_op_name(op_id)
            
            if op_id == OpID.CONST_ZERO:
                # Constant zero
                reg = self._new_register()
                self._instructions.append(SLPInstruction(
                    SLPOpType.CONST_ZERO,
                    dest=reg
                ))
                self._register_map[handle] = reg
                continue
            
            if op_id == OpID.CONST_ONE:
                # Constant one
                reg = self._new_register()
                self._instructions.append(SLPInstruction(
                    SLPOpType.CONST_ONE,
                    dest=reg
                ))
                self._register_map[handle] = reg
                continue
            
            if op_id == OpID.IDENTITY:
                # Identity (variable)
                reg = self._new_register()
                # IDENTITY is a placeholder for the input
                self._instructions.append(SLPInstruction(
                    SLPOpType.IDENTITY,
                    arg1=-1,  # Special marker for input
                    dest=reg
                ))
                self._register_map[handle] = reg
                continue
            
            # Get children registers
            children = node.get_children()
            child_regs = []
            for child in children:
                if child is not None and child in self._register_map:
                    child_regs.append(self._register_map[child])
                elif child is not None:
                    # Child not processed yet - this shouldn't happen in postorder
                    raise CompilationError(f"Child {child} not processed")
            
            # Map op to SLP op type
            slp_op = self._op_id_to_slp_op(op_id)
            
            # Create instruction
            reg = self._new_register()
            
            if len(child_regs) == 0:
                self._instructions.append(SLPInstruction(
                    slp_op,
                    dest=reg
                ))
            elif len(child_regs) == 1:
                self._instructions.append(SLPInstruction(
                    slp_op,
                    arg1=child_regs[0],
                    dest=reg
                ))
            elif len(child_regs) == 2:
                self._instructions.append(SLPInstruction(
                    slp_op,
                    arg1=child_regs[0],
                    arg2=child_regs[1],
                    dest=reg
                ))
            else:
                # For ternary ops like WHERE
                # Currently we only support binary ops, so this shouldn't happen
                raise CompilationError(f"Unsupported arity: {len(child_regs)}")
            
            self._register_map[handle] = reg
        
        # The last register is the output
        if self._handle in self._register_map:
            self._output_register = self._register_map[self._handle]
        else:
            raise CompilationError("Output handle not found in register map")
        
        self._compiled = True
    
    def _op_id_to_slp_op(self, op_id: OpID) -> SLPOpType:
        """Map OpID to SLP operation type."""
        mapping = {
            OpID.ADD: SLPOpType.ADD,
            OpID.SUB: SLPOpType.SUB,
            OpID.MUL: SLPOpType.MUL,
            OpID.DIV: SLPOpType.DIV,
            OpID.POWER: SLPOpType.POW,
            OpID.NEG: SLPOpType.NEG,
            OpID.EXP: SLPOpType.EXP,
            OpID.LOG: SLPOpType.LOG,
            OpID.SIN: SLPOpType.SIN,
            OpID.COS: SLPOpType.COS,
            OpID.TAN: SLPOpType.TAN,
            OpID.TANH: SLPOpType.TANH,
            OpID.ASIN: SLPOpType.ASIN,
            OpID.ACOS: SLPOpType.ACOS,
            OpID.ATAN: SLPOpType.ATAN,
            OpID.SIGMOID: SLPOpType.SIGMOID,
            OpID.RELU: SLPOpType.RELU,
            OpID.SQUARE: SLPOpType.SQUARE,
            OpID.CUBE: SLPOpType.CUBE,
            OpID.SQRT: SLPOpType.SQRT,
            OpID.CBRT: SLPOpType.CBRT,
            OpID.ABS: SLPOpType.ABS,
            OpID.INVERSE: SLPOpType.INVERSE,
            OpID.ERF: SLPOpType.ERF,
            OpID.EXP2: SLPOpType.EXP2,
            OpID.LOG2: SLPOpType.LOG2,
            OpID.WHERE: SLPOpType.WHERE,
            OpID.GREATER: SLPOpType.GREATER,
            OpID.LESS: SLPOpType.LESS,
            OpID.EQUAL: SLPOpType.EQUAL,
            OpID.STOP_GRADIENT: SLPOpType.STOP_GRADIENT,
            OpID.CONST_ZERO: SLPOpType.CONST_ZERO,
            OpID.CONST_ONE: SLPOpType.CONST_ONE,
            OpID.IDENTITY: SLPOpType.IDENTITY,
            OpID.SUM: SLPOpType.SUM,
            OpID.MEAN: SLPOpType.MEAN,
            OpID.VAR: SLPOpType.VAR,
            OpID.STD: SLPOpType.STD,
            OpID.MAX: SLPOpType.MAX,
            OpID.MIN: SLPOpType.MIN,
            OpID.DERIVATIVE: SLPOpType.DERIVATIVE,
            OpID.LEAKY_RELU: SLPOpType.LEAKY_RELU,
            OpID.GELU: SLPOpType.GELU,
            OpID.SWISH: SLPOpType.SWISH,
            OpID.SOFTPLUS: SLPOpType.SOFTPLUS,
        }
        
        if op_id not in mapping:
            raise CompilationError(f"No SLP mapping for op {get_op_name(op_id)}")
        
        return mapping[op_id]
    
    def _new_register(self) -> int:
        """Allocate a new register."""
        self._max_register += 1
        return self._max_register
    
    def get_instructions(self) -> List[SLPInstruction]:
        """Get the list of instructions."""
        return self._instructions
    
    def get_output_register(self) -> int:
        """Get the output register."""
        return self._output_register
    
    def get_register_count(self) -> int:
        """Get the number of registers used."""
        return self._max_register + 1
    
    def get_instruction_count(self) -> int:
        """Get the number of instructions."""
        return len(self._instructions)
    
    def optimize(self) -> None:
        """
        Optimize the straight-line program.
        Performs common subexpression elimination and dead code removal.
        """
        # TODO: Implement optimization passes
        pass
    
    def evaluate(self, inputs: List[float]) -> float:
        """
        Evaluate the program with given inputs.
        
        Args:
            inputs: List of input values (one per variable).
            
        Returns:
            Result of the program.
        """
        if not self._compiled:
            raise CompilationError("Program not compiled")
        
        # Initialize register file
        registers = [0.0] * self.get_register_count()
        
        # Execute instructions
        for instr in self._instructions:
            if instr.op == SLPOpType.CONST_ZERO:
                registers[instr.dest] = 0.0
            elif instr.op == SLPOpType.CONST_ONE:
                registers[instr.dest] = 1.0
            elif instr.op == SLPOpType.CONST:
                registers[instr.dest] = instr.const_val
            elif instr.op == SLPOpType.IDENTITY:
                # Identity maps to input
                if instr.arg1 == -1:
                    registers[instr.dest] = inputs[0]
                else:
                    registers[instr.dest] = registers[instr.arg1]
            elif instr.arg2 is not None:
                # Binary operations
                a = registers[instr.arg1]
                b = registers[instr.arg2]
                
                if instr.op == SLPOpType.ADD:
                    registers[instr.dest] = a + b
                elif instr.op == SLPOpType.SUB:
                    registers[instr.dest] = a - b
                elif instr.op == SLPOpType.MUL:
                    registers[instr.dest] = a * b
                elif instr.op == SLPOpType.DIV:
                    registers[instr.dest] = a / b if b != 0 else float('inf')
                elif instr.op == SLPOpType.POW:
                    registers[instr.dest] = a ** b
                elif instr.op == SLPOpType.WHERE:
                    # WHERE is ternary, but we handle it specially
                    # It should be expanded in the compiler
                    registers[instr.dest] = b if a > 0 else registers.get(instr.metadata.get('arg3', 0), 0.0)
                elif instr.op == SLPOpType.GREATER:
                    registers[instr.dest] = 1.0 if a > b else 0.0
                elif instr.op == SLPOpType.LESS:
                    registers[instr.dest] = 1.0 if a < b else 0.0
                elif instr.op == SLPOpType.EQUAL:
                    registers[instr.dest] = 1.0 if a == b else 0.0
                else:
                    raise CompilationError(f"Unsupported binary op: {instr.op}")
            else:
                # Unary operations
                a = registers[instr.arg1] if instr.arg1 is not None else 0.0
                
                if instr.op == SLPOpType.NEG:
                    registers[instr.dest] = -a
                elif instr.op == SLPOpType.EXP:
                    registers[instr.dest] = math.exp(a)
                elif instr.op == SLPOpType.LOG:
                    registers[instr.dest] = math.log(a) if a > 0 else float('nan')
                elif instr.op == SLPOpType.SIN:
                    registers[instr.dest] = math.sin(a)
                elif instr.op == SLPOpType.COS:
                    registers[instr.dest] = math.cos(a)
                elif instr.op == SLPOpType.TAN:
                    registers[instr.dest] = math.tan(a)
                elif instr.op == SLPOpType.TANH:
                    registers[instr.dest] = math.tanh(a)
                elif instr.op == SLPOpType.ASIN:
                    registers[instr.dest] = math.asin(a) if -1 <= a <= 1 else float('nan')
                elif instr.op == SLPOpType.ACOS:
                    registers[instr.dest] = math.acos(a) if -1 <= a <= 1 else float('nan')
                elif instr.op == SLPOpType.ATAN:
                    registers[instr.dest] = math.atan(a)
                elif instr.op == SLPOpType.ABS:
                    registers[instr.dest] = abs(a)
                elif instr.op == SLPOpType.SQRT:
                    registers[instr.dest] = math.sqrt(a) if a >= 0 else float('nan')
                elif instr.op == SLPOpType.CBRT:
                    registers[instr.dest] = math.copysign(abs(a) ** (1/3), a)
                elif instr.op == SLPOpType.INVERSE:
                    registers[instr.dest] = 1.0 / a if a != 0 else float('inf')
                elif instr.op == SLPOpType.SQUARE:
                    registers[instr.dest] = a * a
                elif instr.op == SLPOpType.CUBE:
                    registers[instr.dest] = a * a * a
                elif instr.op == SLPOpType.ERF:
                    # Approximate erf
                    registers[instr.dest] = math.erf(a)
                elif instr.op == SLPOpType.SIGMOID:
                    registers[instr.dest] = 1.0 / (1.0 + math.exp(-a))
                elif instr.op == SLPOpType.RELU:
                    registers[instr.dest] = max(0.0, a)
                elif instr.op == SLPOpType.LEAKY_RELU:
                    registers[instr.dest] = a if a > 0 else 0.01 * a
                elif instr.op == SLPOpType.SOFTPLUS:
                    registers[instr.dest] = math.log(1.0 + math.exp(a))
                elif instr.op == SLPOpType.EXP2:
                    registers[instr.dest] = 2.0 ** a
                elif instr.op == SLPOpType.LOG2:
                    registers[instr.dest] = math.log2(a) if a > 0 else float('nan')
                elif instr.op == SLPOpType.STOP_GRADIENT:
                    registers[instr.dest] = a
                elif instr.op == SLPOpType.SUM:
                    # Sum of all elements (simplified)
                    registers[instr.dest] = a  # Placeholder
                elif instr.op == SLPOpType.MEAN:
                    registers[instr.dest] = a  # Placeholder
                else:
                    raise CompilationError(f"Unsupported unary op: {instr.op}")
        
        return registers[self._output_register]
    
    def evaluate_batch(self, inputs: List[List[float]]) -> List[float]:
        """
        Evaluate the program on a batch of inputs.
        
        Args:
            inputs: List of input vectors.
            
        Returns:
            List of results.
        """
        return [self.evaluate(x) for x in inputs]
    
    def __repr__(self) -> str:
        """String representation."""
        if not self._compiled:
            return "SLP(not compiled)"
        
        lines = []
        for i, instr in enumerate(self._instructions):
            lines.append(f"  {i:3d}: {instr}")
        lines.append(f"Output: reg{self._output_register}")
        
        return "\n".join([
            f"StraightLineProgram(handle={self._handle})",
            f"Instructions: {len(self._instructions)}",
            f"Registers: {self.get_register_count()}",
            "Instructions:",
            *lines
        ])