"""
Selection operators for genetic programming.
"""

from typing import List, Optional, Tuple, Dict, Any, Callable
import random
import math

from .individual import GPIndividual


class SelectionOperator:
    """
    Base class for selection operators.
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Initialize the selection operator.
        
        Args:
            tournament_size: Size of tournament for tournament selection.
        """
        self.tournament_size = tournament_size
    
    def select(self, population: List[GPIndividual], count: int) -> List[GPIndividual]:
        """
        Select individuals from the population.
        
        Args:
            population: Population to select from.
            count: Number of individuals to select.
            
        Returns:
            List of selected individuals.
        """
        raise NotImplementedError("Subclasses must implement select()")


class TournamentSelection(SelectionOperator):
    """
    Tournament selection operator.
    Selects the best individual from random tournaments.
    """
    
    def __init__(self, tournament_size: int = 3, minimize: bool = True):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Size of tournament.
            minimize: True if minimizing fitness, False if maximizing.
        """
        super().__init__(tournament_size)
        self.minimize = minimize
    
    def select(self, population: List[GPIndividual], count: int) -> List[GPIndividual]:
        """Select individuals using tournament selection."""
        selected = []
        
        for _ in range(count):
            # Select random individuals for tournament
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            
            # Find best in tournament
            if self.minimize:
                best = min(tournament, key=lambda ind: ind.fitness)
            else:
                best = max(tournament, key=lambda ind: ind.fitness)
            
            selected.append(best.copy())
        
        return selected


class RouletteSelection(SelectionOperator):
    """
    Roulette wheel selection (fitness proportionate selection).
    """
    
    def __init__(self, minimize: bool = True):
        """
        Initialize roulette selection.
        
        Args:
            minimize: True if minimizing fitness, False if maximizing.
        """
        super().__init__(1)
        self.minimize = minimize
    
    def select(self, population: List[GPIndividual], count: int) -> List[GPIndividual]:
        """Select individuals using roulette wheel selection."""
        # Convert fitness to selection probability
        if self.minimize:
            # Invert fitness for minimization
            fitness_values = [1.0 / (ind.fitness + 1e-10) for ind in population]
        else:
            fitness_values = [ind.fitness for ind in population]
        
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            # All fitnesses are zero, select randomly
            return random.sample(population, count)
        
        # Compute cumulative probabilities
        probabilities = [f / total_fitness for f in fitness_values]
        cumulative = []
        cum = 0.0
        for p in probabilities:
            cum += p
            cumulative.append(cum)
        
        # Select individuals
        selected = []
        for _ in range(count):
            r = random.random()
            for i, cum_prob in enumerate(cumulative):
                if r <= cum_prob:
                    selected.append(population[i].copy())
                    break
        
        return selected


class RankSelection(SelectionOperator):
    """
    Rank-based selection (linear ranking).
    """
    
    def __init__(self, minimize: bool = True):
        """
        Initialize rank selection.
        
        Args:
            minimize: True if minimizing fitness, False if maximizing.
        """
        super().__init__(1)
        self.minimize = minimize
    
    def select(self, population: List[GPIndividual], count: int) -> List[GPIndividual]:
        """Select individuals using rank-based selection."""
        # Sort by fitness
        if self.minimize:
            sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        else:
            sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        
        # Assign ranks (1 for best)
        n = len(sorted_pop)
        ranks = list(range(n, 0, -1))
        total_ranks = sum(ranks)
        
        # Compute probabilities
        probabilities = [r / total_ranks for r in ranks]
        cumulative = []
        cum = 0.0
        for p in probabilities:
            cum += p
            cumulative.append(cum)
        
        # Select individuals
        selected = []
        for _ in range(count):
            r = random.random()
            for i, cum_prob in enumerate(cumulative):
                if r <= cum_prob:
                    selected.append(sorted_pop[i].copy())
                    break
        
        return selected


class ElitismSelection:
    """
    Elitism selection - preserves the best individuals.
    """
    
    def __init__(self, elitism_count: int, minimize: bool = True):
        """
        Initialize elitism selection.
        
        Args:
            elitism_count: Number of best individuals to preserve.
            minimize: True if minimizing fitness, False if maximizing.
        """
        self.elitism_count = elitism_count
        self.minimize = minimize
    
    def select_elite(self, population: List[GPIndividual]) -> List[GPIndividual]:
        """Select the elite individuals."""
        # Sort by fitness
        if self.minimize:
            sorted_pop = sorted(population, key=lambda ind: ind.fitness)
        else:
            sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
        
        # Return the best individuals
        return [ind.copy() for ind in sorted_pop[:self.elitism_count]]


class StochasticUniversalSampling(SelectionOperator):
    """
    Stochastic Universal Sampling (SUS) selection.
    """
    
    def __init__(self, minimize: bool = True):
        """
        Initialize SUS selection.
        
        Args:
            minimize: True if minimizing fitness, False if maximizing.
        """
        super().__init__(1)
        self.minimize = minimize
    
    def select(self, population: List[GPIndividual], count: int) -> List[GPIndividual]:
        """Select individuals using stochastic universal sampling."""
        # Convert fitness to selection probability
        if self.minimize:
            fitness_values = [1.0 / (ind.fitness + 1e-10) for ind in population]
        else:
            fitness_values = [ind.fitness for ind in population]
        
        total_fitness = sum(fitness_values)
        if total_fitness == 0:
            return random.sample(population, count)
        
        # Compute cumulative probabilities
        probabilities = [f / total_fitness for f in fitness_values]
        cumulative = []
        cum = 0.0
        for p in probabilities:
            cum += p
            cumulative.append(cum)
        
        # Generate equally spaced pointers
        start = random.random() / count
        pointers = [start + i / count for i in range(count)]
        
        # Select individuals
        selected = []
        pointer_idx = 0
        for i, cum_prob in enumerate(cumulative):
            while pointer_idx < count and pointers[pointer_idx] <= cum_prob:
                selected.append(population[i].copy())
                pointer_idx += 1
        
        return selected