"""
Research module for Metanion.
Contains the main GP engine and utilities.
"""

from .metanion_research import (
    print_expr,
    rand_expr,
    collect_nodes,
    mutate_expr,
    crossover_expr,
    evaluate_individual,
    run_island_gp,
    run_gp,
    predict_expression,
    test_expression
)

__all__ = [
    'print_expr',
    'rand_expr',
    'collect_nodes',
    'mutate_expr',
    'crossover_expr',
    'evaluate_individual',
    'run_island_gp',
    'run_gp',
    'predict_expression',
    'test_expression'
]
