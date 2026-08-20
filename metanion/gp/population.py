import random
from .mutation import safe_mutate
from .crossover import safe_crossover
class PopulationManager:
    def __init__(self, population, population_size, evaluator, elitism_count=5, crossover_rate=0.8, mutation_rate=0.3, tournament_size=5, max_depth=5):
        self.population = population[:population_size]
        self.population_size, self.evaluator = population_size, evaluator
        self.elitism_count, self.crossover_rate, self.mutation_rate = elitism_count, crossover_rate, mutation_rate
        self.tournament_size, self.max_depth = tournament_size, max_depth
        self.generation, self.best_individual, self.best_fitness = 0, None, float('inf')
        self.evaluate_population()
    def evaluate_population(self):
        for ind in self.population:
            self.evaluator.evaluate(ind)
            if ind.fitness < self.best_fitness:
                self.best_fitness, self.best_individual = ind.fitness, ind.copy()
    def _tournament_select(self, count=2):
        selected = []
        for _ in range(count):
            tournament = random.sample(self.population, min(self.tournament_size, len(self.population)))
            selected.append(min(tournament, key=lambda x: x.fitness))
        return selected
    def _evolve_one_generation(self):
        sorted_pop = sorted(self.population, key=lambda x: x.fitness)
        new_pop = [ind.copy() for ind in sorted_pop[:self.elitism_count]]
        while len(new_pop) < self.population_size:
            p1, p2 = self._tournament_select(2)
            if random.random() < self.crossover_rate:
                c1, c2 = safe_crossover(p1, p2)
            else:
                c1, c2 = p1.copy(), p2.copy()
            if random.random() < self.mutation_rate:
                c1 = safe_mutate(c1, self.max_depth)
            if random.random() < self.mutation_rate:
                c2 = safe_mutate(c2, self.max_depth)
            new_pop.extend([c1, c2])
        self.population = new_pop[:self.population_size]
        self.generation += 1
        self.evaluate_population()
    def get_best(self):
        return self.best_individual
    def get_stats(self):
        return {'best_fitness': self.best_fitness}
