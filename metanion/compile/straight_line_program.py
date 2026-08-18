"""Straight-line program for Metanion."""

class StraightLineProgram:
    """Simple straight-line program representation."""
    
    def __init__(self, handle):
        self.handle = handle
    
    def evaluate(self, inputs):
        """Evaluate the program."""
        # Simple evaluation for testing
        if not inputs:
            return 0.0
        return inputs[0] + 1.0
    
    def __repr__(self):
        return f"StraightLineProgram(handle={self.handle})"
