import random
def safe_crossover(p1, p2):
    c1, c2 = p1.copy(), p2.copy()
    if random.random() < 0.7 and c1.weight_handles and c2.weight_handles:
        i, j = random.randint(0, len(c1.weight_handles)-1), random.randint(0, len(c2.weight_handles)-1)
        c1.weight_handles[i], c2.weight_handles[j] = c2.weight_handles[j], c1.weight_handles[i]
    return c1, c2
