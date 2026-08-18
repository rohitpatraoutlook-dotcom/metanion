"""
metanion_research.py - Default Island GP Engine

This is the main research module for Metanion.
It uses Island Genetic Programming with constant optimization
for symbolic regression.

Usage:
    from metanion_research import run_gp, print_expr, test_expression
    best = run_gp(X, y)
    print(print_expr(best.weight_handles[0]))
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


# ============================================================================
# 1. EXPRESSION PRINTING
# ============================================================================

def print_expr(handle, var_names=None):
    """
    Convert a handle to a human-readable expression string.

    Args:
        handle: Expression handle.
        var_names: List of variable names (default: ["x0", "x1", ...])

    Returns:
        String representation of the expression.
    """
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
        return f"{on}({l}, {r})"
    return f"{on}(...)"


# ============================================================================
# 2. RANDOM EXPRESSION GENERATION
# ============================================================================

def rand_expr(max_depth=4, n_features=1, add_bias=True):
    """
    Generate a random expression tree.

    Args:
        max_depth: Maximum depth of the tree.
        n_features: Number of input features.
        add_bias: If True, add +1 at the root.

    Returns:
        Handle of the generated expression.
    """
    ops = [OpID.ADD, OpID.SUB, OpID.MUL, OpID.DIV,
           OpID.CONST, OpID.VAR, OpID.SIN, OpID.COS, OpID.EXP, OpID.LOG,
           OpID.SQUARE, OpID.SQRT]

    def build(d):
        if d >= max_depth or random.random() < 0.3:
            choice = random.choice([OpID.CONST, OpID.VAR, OpID.CONST_ZERO, OpID.CONST_ONE])
            if choice == OpID.CONST:
                return intern(choice, value=random.uniform(-3, 3))
            elif choice == OpID.VAR:
                return intern(choice, index=random.randint(0, max(0, n_features - 1)))
            return intern(choice)

        op = random.choice(ops)
        if op in [OpID.ADD, OpID.SUB, OpID.MUL, OpID.DIV]:
            return intern(op, build(d + 1), build(d + 1))
        elif op in [OpID.SIN, OpID.COS, OpID.EXP, OpID.LOG, OpID.SQUARE, OpID.SQRT]:
            return intern(op, build(d + 1))
        elif op == OpID.CONST:
            return intern(op, value=random.uniform(-3, 3))
        elif op == OpID.VAR:
            return intern(op, index=random.randint(0, max(0, n_features - 1)))
        return intern(op)

    expr = build(0)
    if add_bias:
        return intern(OpID.ADD, expr, intern(OpID.CONST_ONE))
    return expr


# ============================================================================
# 3. TREE OPERATIONS
# ============================================================================

def collect_nodes(handle):
    """Collect all nodes in an expression tree."""
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


def mutate_expr(handle, max_depth=4, n_features=1, add_bias=True):
    """Mutate an expression by replacing a subtree."""
    if random.random() < 0.2 or handle is None:
        return rand_expr(max_depth, n_features, add_bias)

    nodes = collect_nodes(handle)
    if not nodes:
        return rand_expr(max_depth, n_features, add_bias)

    target = random.choice(nodes)
    new_expr = rand_expr(max_depth, n_features, False)

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
            return intern(op, child) if child is not None else h
        elif len(node) == 3:
            left = replace(node[1])
            right = replace(node[2])
            if left is not None and right is not None:
                return intern(op, left, right)
            return h
        return h

    result = replace(handle)
    if result is None:
        result = rand_expr(max_depth, n_features, add_bias)
    elif add_bias:
        result = intern(OpID.ADD, result, intern(OpID.CONST_ONE))
    return result


def crossover_expr(h1, h2):
    """Crossover between two expressions."""
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
            return intern(op, child) if child is not None else h
        elif len(node) == 3:
            left = replace(node[1], target, new_sub)
            right = replace(node[2], target, new_sub)
            if left is not None and right is not None:
                return intern(op, left, right)
            return h
        return h

    return replace(h1, t1, t2), replace(h2, t2, t1)


# ============================================================================
# 4. FITNESS EVALUATION
# ============================================================================

def evaluate_individual(ind, X, y, n_features, parsimony_weight=0.001, node_penalty=0.0001):
    """
    Evaluate fitness of an individual.

    Args:
        ind: GPIndividual instance.
        X: Input data.
        y: Target data.
        n_features: Number of input features.
        parsimony_weight: Penalty per depth level.
        node_penalty: Penalty per node.

    Returns:
        Fitness value (lower is better). inf if invalid.
    """
    if ind is None or ind.weight_handles is None or len(ind.weight_handles) == 0:
        return float('inf')

    try:
        f = compile_handle(ind.weight_handles[0], n_features=n_features)
        p = np.array([f(list(x)) for x in X]).flatten()

        if np.any(np.isnan(p)) or np.any(np.isinf(p)):
            return float('inf')

        mse = np.mean((p - y.flatten()) ** 2)
        if mse > 1e6:
            return float('inf')

        depth = get_depth(ind.weight_handles[0], lookup)
        nodes = count_nodes_in_subtree(ind.weight_handles[0], lookup)

        if depth > 10 or nodes > 40:
            return 1e6 + mse

        return mse + parsimony_weight * depth + node_penalty * nodes

    except Exception:
        return float('inf')


# ============================================================================
# 5. CONSTANT OPTIMIZATION
# ============================================================================

def optimize_constants(handle, X, y):
    """Optimize constants in an expression using local search."""
    constants = []

    def collect(h):
        if h is None:
            return
        node = lookup(h)
        if node is None:
            return
        if len(node) >= 2 and node[0] == OpID.CONST:
            constants.append((h, node[1]))
        else:
            if len(node) >= 2 and node[1] is not None:
                collect(node[1])
            if len(node) >= 3 and node[2] is not None:
                collect(node[2])

    collect(handle)

    if not constants:
        return handle

    def evaluate_with_constants(vals):
        val_map = {h: v for h, _ in zip([h for h, _ in constants], vals)}

        def replace(h):
            if h is None:
                return None
            node = lookup(h)
            if node is None:
                return None
            op = node[0]
            if op == OpID.CONST and h in val_map:
                return intern(op, value=val_map[h])
            elif len(node) == 1:
                return h
            elif len(node) == 2:
                child = replace(node[1])
                return intern(op, child) if child is not None else h
            elif len(node) == 3:
                left = replace(node[1])
                right = replace(node[2])
                return intern(op, left, right)
            return h

        new_handle = replace(handle)
        if new_handle is None:
            return 1e10
        try:
            f = compile_handle(new_handle, n_features=X.shape[1])
            pred = np.array([f(list(x)) for x in X]).flatten()
            if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                return 1e10
            return np.mean((pred - y.flatten()) ** 2)
        except:
            return 1e10

    best_vals = [v for _, v in constants]
    best_mse = evaluate_with_constants(best_vals)

    # Try to improve each constant
    for idx in range(len(best_vals)):
        current = best_vals[idx]
        for factor in [0.5, 0.8, 1.2, 1.5, 2.0]:
            new_vals = best_vals[:]
            new_vals[idx] = current * factor
            new_mse = evaluate_with_constants(new_vals)
            if new_mse < best_mse:
                best_vals = new_vals
                best_mse = new_mse

    val_map = {h: v for h, v in zip([h for h, _ in constants], best_vals)}

    def replace(h):
        if h is None:
            return None
        node = lookup(h)
        if node is None:
            return None
        op = node[0]
        if op == OpID.CONST and h in val_map:
            return intern(op, value=val_map[h])
        elif len(node) == 1:
            return h
        elif len(node) == 2:
            child = replace(node[1])
            return intern(op, child) if child is not None else h
        elif len(node) == 3:
            left = replace(node[1])
            right = replace(node[2])
            return intern(op, left, right)
        return h

    return replace(handle)


# ============================================================================
# 6. ISLAND GP (DEFAULT ENGINE)
# ============================================================================

def run_gp(X, y,
           pop_size=120,
           generations=50,
           max_depth=4,
           n_islands=3,
           island_pop=50,
           migration_interval=10,
           migration_count=5,
           add_bias=True,
           optimize_constants=True,
           verbose=False,
           random_seed=None):
    """
    Run Island Genetic Programming for symbolic regression.

    Args:
        X: Input data (n_samples, n_features).
        y: Target data (n_samples, 1).
        pop_size: Total population size (divided across islands).
        generations: Number of generations to evolve.
        max_depth: Maximum depth of expression trees.
        n_islands: Number of islands.
        island_pop: Population per island (overrides pop_size if set).
        migration_interval: Generations between migrations.
        migration_count: Number of individuals to migrate.
        add_bias: If True, add +1 constant to all expressions.
        optimize_constants: If True, optimize constants after GP.
        verbose: If True, print progress.
        random_seed: Random seed for reproducibility.

    Returns:
        Best GPIndividual found.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        random.seed(random_seed)

    n_features = X.shape[1]
    island_pop = island_pop if island_pop else max(30, pop_size // n_islands)

    if verbose:
        print(f"ISLAND GP: {n_islands} islands, {island_pop} each, {generations} gen, features={n_features}")

    # Initialize islands
    islands = []
    for _ in range(n_islands):
        pop = []
        for _ in range(island_pop):
            expr = rand_expr(max_depth, n_features, add_bias)
            ind = GPIndividual(weight_handles=[expr], bias_handle=None, shape=(1, 1))
            ind.fitness = evaluate_individual(ind, X, y, n_features)
            pop.append(ind)
        islands.append(pop)

    best_overall = None
    best_fitness = float('inf')

    for gen in range(generations):
        # Evolve each island
        for island_idx, pop in enumerate(islands):
            pop.sort(key=lambda i: i.fitness)

            # Update best
            if pop[0].fitness < best_fitness:
                best_fitness = pop[0].fitness
                best_overall = pop[0].copy()
                if optimize_constants and best_overall is not None:
                    try:
                        optimized_handle = optimize_constants(best_overall.weight_handles[0], X, y)
                        if optimized_handle is not None:
                            best_overall.weight_handles[0] = optimized_handle
                            best_overall.fitness = evaluate_individual(best_overall, X, y, n_features)
                            best_fitness = best_overall.fitness
                    except:
                        pass

            new_pop = [pop[0].copy()]  # Elitism

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
                    c1_expr = mutate_expr(c1_expr, max_depth, n_features, add_bias)
                if random.random() < 0.4:
                    c2_expr = mutate_expr(c2_expr, max_depth, n_features, add_bias)

                c1 = GPIndividual(weight_handles=[c1_expr], bias_handle=None, shape=(1, 1))
                c2 = GPIndividual(weight_handles=[c2_expr], bias_handle=None, shape=(1, 1))
                c1.fitness = evaluate_individual(c1, X, y, n_features)
                c2.fitness = evaluate_individual(c2, X, y, n_features)

                new_pop.append(c1)
                if len(new_pop) < island_pop:
                    new_pop.append(c2)

            islands[island_idx] = new_pop

        # Migration
        if (gen + 1) % migration_interval == 0 and gen > 0:
            migrants = []
            for pop in islands:
                pop.sort(key=lambda i: i.fitness)
                for i in range(min(migration_count, len(pop))):
                    migrants.append(pop[i].copy())

            for pop in islands:
                for migrant in migrants[:migration_count]:
                    if random.random() < 0.5:
                        pop[random.randint(0, len(pop) - 1)] = migrant.copy()

        if verbose and (gen + 1) % 10 == 0:
            print(f"  Gen {gen+1}: best fitness = {best_fitness:.4f}")

    # Final optimization of best
    if best_overall is not None and optimize_constants:
        try:
            optimized_handle = optimize_constants(best_overall.weight_handles[0], X, y)
            if optimized_handle is not None:
                best_overall.weight_handles[0] = optimized_handle
                best_overall.fitness = evaluate_individual(best_overall, X, y, n_features)
        except:
            pass

    return best_overall


# ============================================================================
# 7. PREDICTION HELPERS
# ============================================================================

def predict_expression(handle, X):
    """Predict using a compiled expression."""
    n_features = X.shape[1]
    f = compile_handle(handle, n_features=n_features)
    return np.array([f(list(x)) for x in X]).flatten()


def test_expression(handle, X_test, y_true):
    """Test an expression and return MSE."""
    y_pred = predict_expression(handle, X_test)
    mse = np.mean((y_pred - y_true.flatten()) ** 2)
    return mse, y_pred


# ============================================================================
# 8. EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("METANION RESEARCH - ISLAND GP EXAMPLE")
    print("=" * 60)

    # Generate data: y = 2*x0 + 3*x1 - x2 + 5
    np.random.seed(42)
    X = np.random.uniform(-5, 5, (200, 3))
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 5 + 0.1 * np.random.randn(200)

    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print("True function: y = 2*x0 + 3*x1 - x2 + 5")

    # Run GP
    best = run_gp(X, y, pop_size=150, generations=60, max_depth=4,
                  optimize_constants=True, verbose=True, random_seed=42)

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Expression: {print_expr(best.weight_handles[0], ['x0', 'x1', 'x2'])}")
    print(f"Fitness: {best.fitness:.6f}")
    print(f"Depth: {best.depth}, Nodes: {best.node_count}")

    # Test predictions
    test_X = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
    true = 2 * test_X[:, 0] + 3 * test_X[:, 1] - test_X[:, 2] + 5
    mse, preds = test_expression(best.weight_handles[0], test_X, true)

    print("\nPredictions:")
    for i in range(len(test_X)):
        print(f"  Input: {test_X[i]}, True: {true[i]:.2f}, Pred: {preds[i]:.2f}")
    print(f"\nMSE: {mse:.6f}")
