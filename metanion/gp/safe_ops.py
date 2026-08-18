"""
Safe operations for GP to prevent NaN, Inf, and domain errors.
All functions return safe values (0.0 or clipped) instead of crashing.
"""

import math


def safe_div(a, b):
    """
    Safe division: returns 0 if denominator is 0 or very small.
    """
    if abs(b) < 1e-12:
        return 0.0
    return a / b


def safe_log(x):
    """
    Safe natural log: returns 0 if x <= 0.
    """
    if x <= 1e-12:
        return 0.0
    return math.log(x)


def safe_log10(x):
    """
    Safe log10: returns 0 if x <= 0.
    """
    if x <= 1e-12:
        return 0.0
    return math.log10(x)


def safe_sqrt(x):
    """
    Safe square root: returns 0 if x < 0.
    """
    if x < 0:
        return 0.0
    return math.sqrt(x)


def safe_pow(base, exp):
    """
    Safe power: only allows valid operations.
    - If base < 0 and exp is fractional, returns 0
    - If result is NaN or Inf, returns 0
    """
    if base < 0 and abs(exp - round(exp)) > 1e-12:
        return 0.0  # Negative base with fractional exponent
    try:
        result = base ** exp
        if math.isnan(result) or math.isinf(result):
            return 0.0
        return result
    except:
        return 0.0


def safe_sin(x):
    """Safe sine: returns 0 on error."""
    try:
        return math.sin(x)
    except:
        return 0.0


def safe_cos(x):
    """Safe cosine: returns 0 on error."""
    try:
        return math.cos(x)
    except:
        return 0.0


def safe_tan(x):
    """Safe tangent: returns 0 on error or infinity."""
    try:
        result = math.tan(x)
        if math.isinf(result):
            return 0.0
        return result
    except:
        return 0.0


def safe_exp(x):
    """Safe exponential: clips to prevent overflow."""
    try:
        result = math.exp(x)
        if math.isinf(result):
            return 1e6  # Clipped to avoid overflow
        return result
    except:
        return 1.0


def safe_inv(x):
    """Safe inverse: returns 0 if x is 0."""
    if abs(x) < 1e-12:
        return 0.0
    return 1.0 / x


def safe_abs(x):
    """Safe absolute value."""
    try:
        return abs(x)
    except:
        return 0.0


def safe_square(x):
    """Safe square: returns 0 on error."""
    try:
        return x * x
    except:
        return 0.0


# Dictionary mapping OpID to safe function
SAFE_OP_MAP = {
    # These will be filled when OpID is available
}

# To be used by the compiler to wrap operations
SAFE_FUNCTIONS = {
    'div': safe_div,
    'log': safe_log,
    'log10': safe_log10,
    'sqrt': safe_sqrt,
    'pow': safe_pow,
    'sin': safe_sin,
    'cos': safe_cos,
    'tan': safe_tan,
    'exp': safe_exp,
    'inv': safe_inv,
    'abs': safe_abs,
    'square': safe_square,
}
