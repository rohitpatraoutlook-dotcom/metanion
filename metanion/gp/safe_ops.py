"""
Safe operations for Metanion - Prevents division by zero and invalid math.
"""

import math


def safe_div(a, b):
    """Safe division: returns 0 if denominator is 0."""
    if abs(b) < 1e-12:
        return 0.0
    return a / b


def safe_log(x):
    """Safe natural log: returns 0 if x <= 0."""
    if x <= 1e-12:
        return 0.0
    return math.log(x)


def safe_log10(x):
    """Safe log10: returns 0 if x <= 0."""
    if x <= 1e-12:
        return 0.0
    return math.log10(x)


def safe_sqrt(x):
    """Safe square root: returns 0 if x < 0."""
    if x < 0:
        return 0.0
    return math.sqrt(x)


def safe_pow(base, exp):
    """Safe power: returns 0 on invalid operations."""
    if base < 0 and abs(exp - round(exp)) > 1e-12:
        return 0.0
    try:
        result = base ** exp
        if math.isnan(result) or math.isinf(result):
            return 0.0
        return result
    except:
        return 0.0


def safe_sin(x):
    try:
        return math.sin(x)
    except:
        return 0.0


def safe_cos(x):
    try:
        return math.cos(x)
    except:
        return 0.0


def safe_tan(x):
    try:
        result = math.tan(x)
        if math.isinf(result):
            return 0.0
        return result
    except:
        return 0.0


def safe_exp(x):
    try:
        result = math.exp(x)
        if math.isinf(result):
            return 1e6
        return result
    except:
        return 1.0


def safe_inv(x):
    """Safe inverse: returns 0 if x is 0."""
    if abs(x) < 1e-12:
        return 0.0
    return 1.0 / x


def safe_abs(x):
    try:
        return abs(x)
    except:
        return 0.0


def safe_square(x):
    try:
        return x * x
    except:
        return 0.0


def safe_cube(x):
    try:
        return x * x * x
    except:
        return 0.0


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
    'cube': safe_cube,
}
