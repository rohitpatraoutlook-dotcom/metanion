"""
metanion_research.py
Research tool for Metanion engine - With POWER operation.
"""

import sys
import os
import numpy as np
import random
from metanion import *
from metanion.symbolic import *
from metanion.compile import *
from metanion.gp.individual import GPIndividual


def print_expr(handle, var="x"):
    if handle is None:
        return "None"
    node = lookup(handle)
    if node is None:
        return "None"
    op = node[0]
    if op == OpID.IDENTITY:
        return var
    elif op == OpID.CONST_ZERO:
        return "0"
    elif op == OpID.CONST_ONE:
        return "1"
    on = get_op_name(op)
    if len(node) == 1:
        return on
    elif len(node) == 2:
        return f"{on}({print_expr(node[1], var)})"
    elif len(node) == 3:
        l = print_expr(node[1], var)
        r = print_expr(node[2], var)
        if on in ['+', '-', '*', '/', '^']:
            return f"({l} {on} {r})"
        return f"{on}({l}, {r})"
    return f"{on}(...)"


def rand_expr(max_depth=3, safe=True, add_bias=True):
    """Include POWER operation for exponents."""
    ops = [OpID.ADD, OpID.SUB, OpID.MUL, OpID.POWER, OpID.IDENTITY,
           OpID.CONST_ZERO, OpID.CONST_ONE]

    def build(d):
        if d >= max_depth or random.random() < 0.4:
            return random.choice([OpID.IDENTITY, OpID.CONST_ZERO, OpID.CONST_ONE])
        op = random.choice(ops)
        if op in [OpID.ADD, OpID.SUB, OpID.MUL, OpID.POWER]:
            return intern(op, build(d+1), build(d+1))
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
                if len(node) >= 2 and node[1] is not None:
                    rec(node[1])
                if len(node) >= 3 and node[2] is not None:
                    rec(node[2])
    rec(handle)
    return nodes


def mutate_expr(handle, max_depth=3, safe=True, add_bias=True):
    nodes = collect_nodes(handle)
    if not nodes or random.random() < 0.3:
        return rand_expr(max_depth, safe, add_bias)

    target = random.choice(nodes)
    new_expr = rand_expr(max_depth, safe, False)

    def replace(h):
        if h == target:
            return new_expr
        node = lookup(h)
        if node is None:
            return h
        op = node[0]
        if len(node) == 1:
            return h
        elif len(node) == 2:
            child = replace(node[1])
            return intern(op, child)
        elif len(node) == 3:
            left = replace(node[1])
            right = replace(node[2])
            return intern(op, left, right)
        return h

    result = replace(handle)
    if add_bias:
        return intern(OpID.ADD, result, intern(OpID.CONST_ONE))
    return result


def crossover_expr(h1, h2):
    n1 = collect_nodes(h1)
    n2 = collect_nodes(h2)
    if not n1 or not n2:
        return h1, h2
    t1 = random.choice(n1)
    t2 = random.choice(n2)

    def replace(h, target, new_sub):
        if h == target:
            return new_sub
        node = lookup(h)
        if node is None:
            return h
        op = node[0]
        if len(node) == 1:
            return h
        elif len(node) == 2:
            child = replace(node[1], target, new_sub)
            return intern(op, child)
        elif len(node) == 3:
            left = replace(node[1], target, new_sub)
            right = replace(node[2], target, new_sub)
            return intern(op, left, right)
        return h

    new_h1 = replace(h1, t1, t2)
    new_h2 = replace(h2, t2, t1)
    return new_h1, new_h2


def evaluate_individual(ind, X, y, parsimony_weight=0.05, node_penalty=0.005):
    try:
        f = compile_handle(ind.weight_handles[0])
        p = np.array([f(float(x[0])) for x in X]).flatten()
        if np.any(np.isnan(p)) or np.any(np.isinf(p)):
            return float('inf')
        mse = np.mean((p - y.flatten())**2)
        depth = get_depth(ind.weight_handles[0], lookup)
        nodes = count_nodes_in_subtree(ind.weight_handles[0], lookup)
        return mse + parsimony_weight * depth + node_penalty * nodes
    except Exception:
        return float('inf')


def run_gp(X, y,
           pop_size=200,
           generations=100,
           max_depth=3,
           safe=True,
           add_bias=True,
           crossover_rate=0.8,
           mutation_rate=0.5,
           parsimony_weight=0.05,
           node_penalty=0.005,
           verbose=True,
           random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    if verbose:
        print(f"GP: pop={pop_size}, gen={generations}, depth={max_depth}, bias={add_bias}")

    pop = [GPIndividual(weight_handles=[rand_expr(max_depth, safe, add_bias)],
                        bias_handle=None, shape=(1, 1)) for _ in range(pop_size)]

    for ind in pop:
        ind.fitness = evaluate_individual(ind, X, y, parsimony_weight, node_penalty)

    best = min(pop, key=lambda i: i.fitness)

    for gen in range(generations):
        pop.sort(key=lambda i: i.fitness)
        new_pop = [pop[0].copy()]

        while len(new_pop) < pop_size:
            t1 = random.sample(pop[:min(30, len(pop))], 3)
            t2 = random.sample(pop[:min(30, len(pop))], 3)
            p1 = min(t1, key=lambda i: i.fitness)
            p2 = min(t2, key=lambda i: i.fitness)

            if random.random() < crossover_rate:
                c1_expr, c2_expr = crossover_expr(p1.weight_handles[0], p2.weight_handles[0])
            else:
                c1_expr, c2_expr = p1.weight_handles[0], p2.weight_handles[0]

            if random.random() < mutation_rate:
                c1_expr = mutate_expr(c1_expr, max_depth, safe, add_bias)
            if random.random() < mutation_rate:
                c2_expr = mutate_expr(c2_expr, max_depth, safe, add_bias)

            c1 = GPIndividual(weight_handles=[c1_expr], bias_handle=None, shape=(1, 1))
            c2 = GPIndividual(weight_handles=[c2_expr], bias_handle=None, shape=(1, 1))
            c1.fitness = evaluate_individual(c1, X, y, parsimony_weight, node_penalty)
            c2.fitness = evaluate_individual(c2, X, y, parsimony_weight, node_penalty)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        pop = new_pop
        best = min(pop, key=lambda i: i.fitness)

        if verbose and (gen + 1) % 20 == 0:
            expr_str = print_expr(best.weight_handles[0])
            if len(expr_str) > 60:
                expr_str = expr_str[:60] + "..."
            print(f"  Gen {gen+1}: fitness={best.fitness:.4f}, expr={expr_str}")

    return best


def predict_expression(handle, X):
    f = compile_handle(handle)
    return np.array([f(float(x)) for x in X.flatten()])


def test_expression(handle, X_test, y_true):
    y_pred = predict_expression(handle, X_test)
    mse = np.mean((y_pred - y_true.flatten())**2)
    return mse, y_pred


if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.uniform(-2, 2, (200, 1))
    y = X**3 - 2*X**2 + X + 1 + 0.1 * np.random.randn(200, 1)

    print("=" * 80)
    print("METANION RESEARCH (WITH POWER OPERATION)")
    print("=" * 80)
    print(f"True: y = x^3 - 2x^2 + x + 1")
    print(f"Samples: {len(X)}")

    best = run_gp(X, y, pop_size=200, generations=100, max_depth=3,
                  safe=True, add_bias=True, verbose=True, random_seed=42)

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Expression: {print_expr(best.weight_handles[0])}")
    print(f"Fitness: {best.fitness:.6f}")
    print(f"Depth: {get_depth(best.weight_handles[0], lookup)}")
    print(f"Nodes: {count_nodes_in_subtree(best.weight_handles[0], lookup)}")

    test_X = np.array([[-1.5], [-0.5], [0.0], [0.5], [1.5]])
    true = test_X**3 - 2*test_X**2 + test_X + 1
    mse, preds = test_expression(best.weight_handles[0], test_X, true)

    print("\nPREDICTIONS")
    print("  x_true | y_true | y_pred | Error")
    for i in range(len(test_X)):
        print(f"  {test_X[i,0]:6.1f} | {true[i,0]:6.1f} | {preds[i]:6.3f} | {abs(preds[i]-true[i,0]):.3f}")
    print(f"\nMSE: {mse:.6f}")

    h = best.weight_handles[0]
    print(f"\nHandle: {h} (type: {type(h)})")
    print(f"Node: {lookup(h)}")
    print("PROOF: Weights are symbolic expressions, not numbers.")
