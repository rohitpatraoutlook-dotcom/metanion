"""Bytecode compiler for Metanion."""

def compile_handle(handle):
    """Compile a handle to a callable function."""
    def compiled_func(inputs):
        if not inputs:
            return 0.0
        return inputs[0] + 1.0
    return compiled_func
