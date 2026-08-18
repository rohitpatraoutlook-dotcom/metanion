"""
JIT bytecode compiler for the Metanion engine.
Compiles straight-line programs to Python bytecode for fast evaluation.
"""

from typing import Optional, List, Tuple, Dict, Any, Callable
import types
import inspect
import dis
import sys

from .straight_line_program import SLPOpType, SLPInstruction, StraightLineProgram
from ..exceptions import CompilationError
from ..config import ACTIVE_CONFIG


class BytecodeCompiler:
    """
    Compiles a straight-line program to executable Python bytecode.
    Uses Python's built-in compile() and exec() for JIT compilation.
    """
    
    def __init__(self):
        """Initialize the bytecode compiler."""
        self._cache: Dict[int, Callable] = {}  # handle -> compiled function
        self._cache_hits = 0
        self._cache_misses = 0
    
    def compile(self, program: StraightLineProgram) -> Callable[[List[float]], float]:
        """
        Compile a straight-line program to a callable function.
        
        Args:
            program: The straight-line program to compile.
            
        Returns:
            A callable function that evaluates the program.
            
        Raises:
            CompilationError: If compilation fails.
        """
        # Check cache
        handle = program._handle
        if handle in self._cache:
            self._cache_hits += 1
            return self._cache[handle]
        
        self._cache_misses += 1
        
        try:
            # Generate Python source code
            source = self._generate_source(program)
            
            # Compile to bytecode
            compiled = compile(source, '<metanion>', 'exec')
            
            # Create a namespace
            namespace = {
                'math': __import__('math'),
                'builtins': __import__('builtins'),
            }
            
            # Execute the compiled code
            exec(compiled, namespace)
            
            # Get the compiled function
            func_name = f"_metanion_eval_{handle}"
            if func_name not in namespace:
                raise CompilationError(f"Function {func_name} not found in compiled namespace")
            
            func = namespace[func_name]
            
            # Cache the compiled function
            if len(self._cache) < ACTIVE_CONFIG.jit_cache_size:
                self._cache[handle] = func
            
            return func
        
        except Exception as e:
            raise CompilationError(f"Failed to compile program: {e}")
    
    def _generate_source(self, program: StraightLineProgram) -> str:
        """
        Generate Python source code from a straight-line program.
        
        Args:
            program: The straight-line program.
            
        Returns:
            Python source code as a string.
        """
        instructions = program.get_instructions()
        num_registers = program.get_register_count()
        output_reg = program.get_output_register()
        handle = program._handle
        
        # Generate function signature and body
        lines = []
        lines.append(f"def _metanion_eval_{handle}(inputs):")
        lines.append(f"    # Input validation")
        lines.append(f"    if len(inputs) < 1:")
        lines.append(f"        raise ValueError('Need at least 1 input')")
        lines.append(f"    # Register file")
        lines.append(f"    reg = [0.0] * {num_registers}")
        lines.append(f"    # Set up math functions for speed")
        lines.append(f"    exp = math.exp")
        lines.append(f"    log = math.log")
        lines.append(f"    sin = math.sin")
        lines.append(f"    cos = math.cos")
        lines.append(f"    tan = math.tan")
        lines.append(f"    tanh = math.tanh")
        lines.append(f"    sqrt = math.sqrt")
        lines.append(f"    erf = math.erf")
        
        # Generate instructions
        for idx, instr in enumerate(instructions):
            line = self._instruction_to_source(instr, idx)
            if line:
                lines.append(f"    # {idx}: {instr}")
                lines.append(f"    {line}")
        
        # Return the result
        lines.append(f"    return reg[{output_reg}]")
        
        return "\n".join(lines)
    
    def _instruction_to_source(self, instr: SLPInstruction, idx: int) -> str:
        """
        Convert an instruction to Python source code.
        
        Args:
            instr: The instruction.
            idx: Instruction index (for debugging).
            
        Returns:
            Python source code line.
        """
        dest = instr.dest
        op = instr.op
        
        if op == SLPOpType.CONST_ZERO:
            return f"reg[{dest}] = 0.0"
        
        if op == SLPOpType.CONST_ONE:
            return f"reg[{dest}] = 1.0"
        
        if op == SLPOpType.CONST:
            return f"reg[{dest}] = {instr.const_val:.15f}"
        
        if op == SLPOpType.IDENTITY:
            if instr.arg1 == -1:
                return f"reg[{dest}] = inputs[0]"
            else:
                return f"reg[{dest}] = reg[{instr.arg1}]"
        
        if instr.arg2 is not None:
            # Binary operations
            arg1 = instr.arg1
            arg2 = instr.arg2
            
            if op == SLPOpType.ADD:
                return f"reg[{dest}] = reg[{arg1}] + reg[{arg2}]"
            elif op == SLPOpType.SUB:
                return f"reg[{dest}] = reg[{arg1}] - reg[{arg2}]"
            elif op == SLPOpType.MUL:
                return f"reg[{dest}] = reg[{arg1}] * reg[{arg2}]"
            elif op == SLPOpType.DIV:
                return f"reg[{dest}] = reg[{arg1}] / reg[{arg2}] if reg[{arg2}] != 0 else float('inf')"
            elif op == SLPOpType.POW:
                return f"reg[{dest}] = reg[{arg1}] ** reg[{arg2}]"
            elif op == SLPOpType.GREATER:
                return f"reg[{dest}] = 1.0 if reg[{arg1}] > reg[{arg2}] else 0.0"
            elif op == SLPOpType.LESS:
                return f"reg[{dest}] = 1.0 if reg[{arg1}] < reg[{arg2}] else 0.0"
            elif op == SLPOpType.EQUAL:
                return f"reg[{dest}] = 1.0 if reg[{arg1}] == reg[{arg2}] else 0.0"
            elif op == SLPOpType.WHERE:
                # WHERE is ternary - handle specially
                return f"reg[{dest}] = reg[{arg2}] if reg[{arg1}] > 0 else 0.0"
            else:
                raise CompilationError(f"Unsupported binary op: {op}")
        else:
            # Unary operations
            arg1 = instr.arg1 if instr.arg1 is not None else -1
            if arg1 == -1:
                src = "inputs[0]"
            else:
                src = f"reg[{arg1}]"
            
            if op == SLPOpType.NEG:
                return f"reg[{dest}] = -{src}"
            elif op == SLPOpType.EXP:
                return f"reg[{dest}] = exp({src})"
            elif op == SLPOpType.LOG:
                return f"reg[{dest}] = log({src}) if {src} > 0 else float('nan')"
            elif op == SLPOpType.SIN:
                return f"reg[{dest}] = sin({src})"
            elif op == SLPOpType.COS:
                return f"reg[{dest}] = cos({src})"
            elif op == SLPOpType.TAN:
                return f"reg[{dest}] = tan({src})"
            elif op == SLPOpType.TANH:
                return f"reg[{dest}] = tanh({src})"
            elif op == SLPOpType.ASIN:
                return f"reg[{dest}] = math.asin({src}) if -1 <= {src} <= 1 else float('nan')"
            elif op == SLPOpType.ACOS:
                return f"reg[{dest}] = math.acos({src}) if -1 <= {src} <= 1 else float('nan')"
            elif op == SLPOpType.ATAN:
                return f"reg[{dest}] = math.atan({src})"
            elif op == SLPOpType.ABS:
                return f"reg[{dest}] = abs({src})"
            elif op == SLPOpType.SQRT:
                return f"reg[{dest}] = sqrt({src}) if {src} >= 0 else float('nan')"
            elif op == SLPOpType.CBRT:
                return f"reg[{dest}] = math.copysign(abs({src}) ** (1/3), {src})"
            elif op == SLPOpType.INVERSE:
                return f"reg[{dest}] = 1.0 / {src} if {src} != 0 else float('inf')"
            elif op == SLPOpType.SQUARE:
                return f"reg[{dest}] = {src} * {src}"
            elif op == SLPOpType.CUBE:
                return f"reg[{dest}] = {src} * {src} * {src}"
            elif op == SLPOpType.ERF:
                return f"reg[{dest}] = erf({src})"
            elif op == SLPOpType.SIGMOID:
                return f"reg[{dest}] = 1.0 / (1.0 + exp(-{src}))"
            elif op == SLPOpType.RELU:
                return f"reg[{dest}] = {src} if {src} > 0 else 0.0"
            elif op == SLPOpType.LEAKY_RELU:
                return f"reg[{dest}] = {src} if {src} > 0 else 0.01 * {src}"
            elif op == SLPOpType.SOFTPLUS:
                return f"reg[{dest}] = log(1.0 + exp({src}))"
            elif op == SLPOpType.EXP2:
                return f"reg[{dest}] = 2.0 ** {src}"
            elif op == SLPOpType.LOG2:
                return f"reg[{dest}] = math.log2({src}) if {src} > 0 else float('nan')"
            elif op == SLPOpType.STOP_GRADIENT:
                return f"reg[{dest}] = {src}"
            else:
                raise CompilationError(f"Unsupported unary op: {op}")
    
    def clear_cache(self) -> None:
        """Clear the compilation cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._cache),
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'hit_ratio': self._cache_hits / (self._cache_hits + self._cache_misses) 
                        if (self._cache_hits + self._cache_misses) > 0 else 0.0,
        }


class OptimizedBytecodeCompiler(BytecodeCompiler):
    """
    Optimized bytecode compiler with additional optimizations.
    """
    
    def _generate_source(self, program: StraightLineProgram) -> str:
        """
        Generate optimized Python source code.
        Performs constant folding and dead code elimination.
        """
        # TODO: Implement constant folding and dead code elimination
        return super()._generate_source(program)
    
    def _optimize_instruction(self, instr: SLPInstruction) -> SLPInstruction:
        """
        Optimize a single instruction.
        
        Performs:
        - Constant folding (e.g., 0 + x -> x)
        - Strength reduction (e.g., x * 2 -> x + x)
        - Identity elimination (e.g., x + 0 -> x)
        """
        # TODO: Implement instruction-level optimizations
        return instr


# Global compiler instance
_COMPILER: Optional[BytecodeCompiler] = None


def get_compiler() -> BytecodeCompiler:
    """Get or create the global bytecode compiler."""
    global _COMPILER
    if _COMPILER is None:
        _COMPILER = BytecodeCompiler()
    return _COMPILER


def compile_program(program: StraightLineProgram) -> Callable[[List[float]], float]:
    """Compile a straight-line program to a callable function."""
    return get_compiler().compile(program)


def compile_handle(handle: int) -> Callable[[List[float]], float]:
    """Compile a handle to a callable function."""
    program = StraightLineProgram(handle)
    return compile_program(program)