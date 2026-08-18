"""
metanion_research.py - Full GP with all operations, constant penalty, and composition.
"""

import sys
import os
import numpy as np
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metanion import *
from metanion.symbolic import *
from metanion.compile import *
from metanion.gp.individual import GPIndividual
from metanion.gp.regularization import SymbolicRegularization
from metanion.gp.safe_ops import *


def print_expr(handle, var_names=None):
    if var_names is None:
        var_names = ["x0", "x1", "x2", "x3", "x4"]
    if handle is None:
        return "None"
    node = lookup(handle)
    if node is None:
        return "None"
    op = node[0]
    if op == OpID.IDENTITY:
        return var_names[0] if var_names else "x"
    elif op == OpID.CONST_ZERO:
        return "0"
    elif op == OpID.CONST_ONE:
        return "1"
    elif op == OpID.CONST:
        return f"{node[1]:.4f}"
    elif op == OpID.VAR:
        idx = node[1]
        return var_names[idx] if idx < len(var_names) else f"x{idx}"
    on = get_op_name(op)
    if len(node) == 1:
        return on
    elif len(node) == 2:
        return f"{on}({print_expr(node[1], var_names)})"
    elif len(node) == 3:
        l = print_expr(node[1], var_names)
        r = print_expr(node[2], var_names)
        if on in ['+', '-', '*', '/']:
            return f"({l} {on} {r})"
        elif on == '^':
            return f"({l} ** {r})"
        return f"{on}({l}, {r})"
    return f"{on}(...)"


def make_const(value=None):
    if value is None:
        value = random.uniform(-5, 5)
    return intern(OpID.CONST, value=value)


def make_var(n_features):
    return intern(OpID.VAR, index=random.randint(0, max(0, n_features-1)))


def is_constant_expression(handle):
    """Check if an expression is just a constant."""
    node = lookup(handle)
    if node is None:
        return True
    op = node[0]
    if op in [OpID.CONST_ZERO, OpID.CONST_ONE, OpID.CONST]:
        return True
    if len(node) == 3:
        left_const = is_constant_expression(node[1])
        right_const = is_constant_expression(node[2])
        if left_const and right_const:
            return True
    return False


def rand_expr(max_depth=4, n_features=1, add_bias=True, allowed_ops=None):
    """Generate random expression with ALL operations."""
    op_map = {
        'add': OpID.ADD, 'sub': OpID.SUB, 'mul': OpID.MUL,
        'div': OpID.DIV, 'const': OpID.CONST, 'var': OpID.VAR,
        'square': OpID.SQUARE, 'cube': OpID.CUBE, 'sqrt': OpID.SQRT,
        'sin': OpID.SIN, 'cos': OpID.COS, 'tan': OpID.TAN,
        'exp': OpID.EXP, 'log': OpID.LOG, 'log10': OpID.LOG10,
        'abs': OpID.ABS, 'inv': OpID.INVERSE
    }
    
    if allowed_ops is None:
        ops = [OpID.ADD, OpID.SUB, OpID.MUL, OpID.DIV,
               OpID.CONST, OpID.VAR, OpID.SQRT, OpID.SQUARE,
               OpID.SIN, OpID.COS, OpID.EXP, OpID.LOG]
    else:
        ops = [op_map[op] for op in allowed_ops if op in op_map]
        if not ops:
            ops = [OpID.ADD, OpID.SUB, OpID.MUL, OpID.CONST, OpID.VAR]
    
    def build(d):
        if d >= max_depth or random.random() < 0.25:
            choice = random.choice([OpID.CONST, OpID.VAR, OpID.CONST_ZERO, OpID.CONST_ONE])
            if choice == OpID.CONST:
                return make_const()
            elif choice == OpID.VAR:
                return make_var(n_features)
            return intern(choice)
        
        op = random.choice(ops)
        if op in [OpID.ADD, OpID.SUB, OpID.MUL, OpID.DIV]:
            return intern(op, build(d+1), build(d+1))
        elif op in [OpID.SIN, OpID.COS, OpID.TAN, OpID.EXP, OpID.LOG, OpID.LOG10,
                    OpID.SQRT, OpID.SQUARE, OpID.CUBE, OpID.ABS, OpID.INVERSE]:
            return intern(op, build(d+1))
        elif op == OpID.CONST:
            return make_const()
        elif op == OpID.VAR:
            return make_var(n_features)
        return intern(op)
    
    expr = build(0)
    if add_bias:
        return intern(OpID.ADD, expr, intern(OpID.CONST_ONE))
    return expr


def collect_nodes(handle):
    nodes = []
    def rec(h):
        if h is not None:
            nodes.append(h)
            node = lookup(h)
            if node is not None:
                if len(node) >= 2 and node[0] in [OpID.CONST, OpID.VAR]:
                    return
                if len(node) >= 2 and node[1] is not None:
                    rec(node[1])
                if len(node) >= 3 and node[2] is not None:
                    rec(node[2])
    rec(handle)
    return nodes


def mutate_expr(handle, max_depth=4, n_features=1, add_bias=True, allowed_ops=None):
    if random.random() < 0.2 or handle is None:
        return rand_expr(max_depth, n_features, add_bias, allowed_ops)
    
    nodes = collect_nodes(handle)
    if not nodes:
        return rand_expr(max_depth, n_features, add_bias, allowed_ops)
    
    target = random.choice(nodes)
    new_expr = rand_expr(max_depth, n_features, False, allowed_ops)

    def replace(h):
        if h == target:
            return new_expr
        if h is None:
            return None
        node = lookup(h)
        if node is None:
            return h
        op = node[0]
        if len(node) == 1:
            return h
        elif len(node) == 2:
            if op in [OpID.CONST, OpID.VAR]:
                return h
            child = replace(node[1])
            if child is not None:
                return intern(op, child)
            return h
        elif len(node) == 3:
            left = replace(node[1])
            right = replace(node[2])
            if left is not None and right is not None:
                return intern(op, left, right)
            return h
        return h

    result = replace(handle)
    if result is None:
        result = rand_expr(max_depth, n_features, add_bias, allowed_ops)
    elif add_bias:
        result = intern(OpID.ADD, result, intern(OpID.CONST_ONE))
    return result


def crossover_expr(h1, h2):
    if h1 is None or h2 is None:
        return h1, h2
    n1 = collect_nodes(h1)
    n2 = collect_nodes(h2)
    if not n1 or not n2:
        return h1, h2
    
    t1 = random.choice(n1)
    t2 = random.choice(n2)

    def replace(h, target, new_sub):
        if h == target:
            return new_sub
        if h is None:
            return None
        node = lookup(h)
        if node is None:
            return h
        op = node[0]
        if len(node) == 1:
            return h
        elif len(node) == 2:
            if op in [OpID.CONST, OpID.VAR]:
                return h
            child = replace(node[1], target, new_sub)
            if child is not None:
                return intern(op, child)
            return h
        elif len(node) == 3:
            left = replace(node[1], target, new_sub)
            right = replace(node[2], target, new_sub)
            if left is not None and right is not None:
                return intern(op, left, right)
            return h
        return h

    return replace(h1, t1, t2), replace(h2, t2, t1)


def evaluate_individual(ind, X, y, n_features, reg_params):
    if ind is None or ind.weight_handles is None or len(ind.weight_handles) == 0:
        return float('inf')
    
    # Reject constants
    if is_constant_expression(ind.weight_handles[0]):
        return float('inf')
    
    try:
        f = compile_handle(ind.weight_handles[0], n_features=n_features)
        p = np.array([f(list(x)) for x in X]).flatten()
        if np.any(np.isnan(p)) or np.any(np.isinf(p)):
            return float('inf')
        
        mse = np.mean((p - y.flatten())**2)
        
        try:
            depth = get_depth(ind.weight_handles[0], lookup)
            nodes = count_nodes_in_subtree(ind.weight_handles[0], lookup)
        except RecursionError:
            return float('inf')
        
        max_depth = reg_params['max_depth']
        complexity_penalty = reg_params['complexity_penalty']
        
        if depth > max_depth:
            return float('inf')
        
        return mse + 0.01 * depth + 0.001 * nodes + complexity_penalty * nodes
        
    except RecursionError:
        return float('inf')
    except Exception:
        return float('inf')


def run_island_gp(X, y,
                  n_islands=3,
                  island_pop=80,
                  generations=60,
                  max_depth=4,
                  add_bias=True,
                  optimize_constants=False,
                  verbose=False,
                  random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)
    
    n_features = X.shape[1]
    
    reg_obj = SymbolicRegularization(X, y)
    reg_params = reg_obj.get_params()
    
    if verbose:
        print(f"ISLAND GP with Symbolic Regularization:")
        print(f"  Data linearity: {reg_params['linearity_score']:.4f}")
        print(f"  Complexity penalty: {reg_params['complexity_penalty']:.4f}")
        print(f"  Max depth: {reg_params['max_depth']}")
        print(f"  Features: {n_features}")
    
    adaptive_depth = reg_params['max_depth']
    allowed_ops = reg_params['allowed_ops']
    
    islands = []
    for _ in range(n_islands):
        pop = []
        for _ in range(island_pop):
            expr = rand_expr(adaptive_depth, n_features, add_bias, allowed_ops)
            ind = GPIndividual(weight_handles=[expr], bias_handle=None, shape=(1, 1))
            ind.fitness = evaluate_individual(ind, X, y, n_features, reg_params)
            pop.append(ind)
        islands.append(pop)
    
    best_overall = None
    best_fitness = float('inf')
    
    for gen in range(generations):
        for island_idx, pop in enumerate(islands):
            pop.sort(key=lambda i: i.fitness)
            
            if pop[0].fitness < best_fitness:
                best_fitness = pop[0].fitness
                best_overall = pop[0].copy()
            
            new_pop = [pop[0].copy()]
            
            while len(new_pop) < island_pop:
                t1 = random.sample(pop[:min(20, len(pop))], 3)
                t2 = random.sample(pop[:min(20, len(pop))], 3)
                p1 = min(t1, key=lambda i: i.fitness)
                p2 = min(t2, key=lambda i: i.fitness)
                
                if random.random() < 0.7:
                    c1_expr, c2_expr = crossover_expr(p1.weight_handles[0], p2.weight_handles[0])
                else:
                    c1_expr, c2_expr = p1.weight_handles[0], p2.weight_handles[0]
                
                if random.random() < 0.4:
                    c1_expr = mutate_expr(c1_expr, adaptive_depth, n_features, add_bias, allowed_ops)
                if random.random() < 0.4:
                    c2_expr = mutate_expr(c2_expr, adaptive_depth, n_features, add_bias, allowed_ops)
                
                c1 = GPIndividual(weight_handles=[c1_expr], bias_handle=None, shape=(1, 1))
                c2 = GPIndividual(weight_handles=[c2_expr], bias_handle=None, shape=(1, 1))
                c1.fitness = evaluate_individual(c1, X, y, n_features, reg_params)
                c2.fitness = evaluate_individual(c2, X, y, n_features, reg_params)
                
                new_pop.append(c1)
                if len(new_pop) < island_pop:
                    new_pop.append(c2)
            
            islands[island_idx] = new_pop
        
        if (gen + 1) % 10 == 0 and gen > 0:
            if verbose:
                print(f"  Gen {gen+1}: best fitness = {best_fitness:.4f}")
    
    return best_overall


def run_gp(X, y,
           pop_size=100,
           generations=50,
           max_depth=4,
           add_bias=True,
           optimize_constants=False,
           verbose=False,
           random_seed=None,
           **kwargs):
    return run_island_gp(
        X, y,
        n_islands=3,
        island_pop=max(50, pop_size // 3),
        generations=generations,
        max_depth=max_depth,
        add_bias=add_bias,
        optimize_constants=optimize_constants,
        verbose=verbose,
        random_seed=random_seed
    )


def predict_expression(handle, X):
    n_features = X.shape[1]
    f = compile_handle(handle, n_features=n_features)
    return np.array([f(list(x)) for x in X]).flatten()


def test_expression(handle, X_test, y_true):
    y_pred = predict_expression(handle, X_test)
    mse = np.mean((y_pred - y_true.flatten())**2)
    return mse, y_pred
