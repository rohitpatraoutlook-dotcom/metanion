"""
Population management for genetic programming.
"""

from typing import List, Optional, Dict, Any, Tuple
import random
from collections import defaultdict

from .individual import GPIndividual
from .initialization import PopulationInitializer
from .fitness import FitnessEvaluator
from .selection import TournamentSelection, ElitismSelection
from .crossover import SubtreeCrossover
from .mutation import PointMutation, SubtreeMutation


class PopulationManager:
    """
    Manages the GP population evolution.
    """
    
    def __init__(
        self,
        population: Optional[List[GPIndividual]] = None,
        population_size: int = 100,
        evaluator: Optional[FitnessEvaluator] = None,
        elitism_count: int = 5,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        max_depth: int = 10
    ):
        """
        Initialize the population manager.
        
        Args:
            population: Initial population (if None, will be initialized).
            population_size: Size of the population.
            evaluator: Fitness evaluator.
            elitism_count: Number of elite individuals to preserve.
            crossover_rate: Probability of crossover.
            mutation_rate: Probability of mutation.
            tournament_size: Size of tournament selection.
            max_depth: Maximum depth of expressions.
        """
        self.population_size = population_size
        self.evaluator = evaluator
        self.elitism_count = elitism_count
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.max_depth = max_depth
        
        # Initialize population
        if population is None:
            self.population = self._initialize_population()
        else:
            self.population = population[:population_size]
        
        # Initialize operators
        self.selection = TournamentSelection(tournament_size)
        self.elitism = ElitismSelection(elitism_count)
        self.crossover = SubtreeCrossover(max_depth)
        self.mutation_point = PointMutation(max_depth)
        self.mutation_subtree = SubtreeMutation(max_depth)
        
        # Track best individual
        self.best_individual = None
        self.best_fitness = float('inf')
        self.generation = 0
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'best_depth': [],
            'avg_depth': [],
        }
    
    def _initialize_population(self) -> List[GPIndividual]:
        """Initialize the population."""
        initializer = PopulationInitializer(
            self.population_size,
            (1, 1),
            self.max_depth
        )
        return initializer.initialize()
    
    def evaluate_population(self) -> None:
        """Evaluate all individuals in the population."""
        if self.evaluator is None:
            return
        
        self.evaluator.evaluate_population(self.population)
        
        # Update best individual
        for ind in self.population:
            if ind.fitness < self.best_fitness:
                self.best_fitness = ind.fitness
                self.best_individual = ind.copy()
    
    def evolve(self, generations: int = 1) -> None:
        """
        Evolve the population for a number of generations.
        
        Args:
            generations: Number of generations to evolve.
        """
        for _ in range(generations):
            self._evolve_one_generation()
    
    def _evolve_one_generation(self) -> None:
        """Evolve one generation."""
        # Evaluate if needed
        if any(ind.fitness == float('inf') for ind in self.population):
            self.evaluate_population()
        
        # Create new population
        new_population = []
        
        # Elitism - preserve best individuals
        elites = self.elitism.select_elite(self.population)
        new_population.extend(elites)
        
        # Generate offspring
        while len(new_population) < self.population_size:
            # Selection
            parents = self.selection.select(self.population, 2)
            parent1, parent2 = parents[0], parents[1] if len(parents) > 1 else parents[0]
            
            # Crossover
            if random.random() < self.crossover_rate:
                offspring1, offspring2 = self.crossover.crossover(parent1, parent2)
            else:
                offspring1, offspring2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < self.mutation_rate:
                if random.random() < 0.5:
                    offspring1 = self.mutation_point.mutate(offspring1)
                else:
                    offspring1 = self.mutation_subtree.mutate(offspring1)
            
            if random.random() < self.mutation_rate:
                if random.random() < 0.5:
                    offspring2 = self.mutation_point.mutate(offspring2)
                else:
                    offspring2 = self.mutation_subtree.mutate(offspring2)
            
            # Update generation
            offspring1.generation = self.generation + 1
            offspring2.generation = self.generation + 1
            
            # Add to new population
            new_population.append(offspring1)
            if len(new_population) < self.population_size:
                new_population.append(offspring2)
        
        self.population = new_population[:self.population_size]
        self.generation += 1
        
        # Evaluate the new population
        self.evaluate_population()
        
        # Update history
        self._update_history()
    
    def _update_history(self) -> None:
        """Update the evolution history."""
        fitnesses = [ind.fitness for ind in self.population if ind.fitness != float('inf')]
        
        if fitnesses:
            avg_fitness = sum(fitnesses) / len(fitnesses)
            depths = [ind.depth for ind in self.population]
            avg_depth = sum(depths) / len(depths)
            
            self.history['best_fitness'].append(self.best_fitness)
            self.history['avg_fitness'].append(avg_fitness)
            self.history['best_depth'].append(self.best_individual.depth if self.best_individual else 0)
            self.history['avg_depth'].append(avg_depth)
    
    def get_best(self) -> Optional[GPIndividual]:
        """Get the best individual found so far."""
        return self.best_individual
    
    def get_population(self) -> List[GPIndividual]:
        """Get the current population."""
        return self.population
    
    def get_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        fitnesses = [ind.fitness for ind in self.population if ind.fitness != float('inf')]
        depths = [ind.depth for ind in self.population]
        node_counts = [ind.node_count for ind in self.population]
        times = [ind.inference_time for ind in self.population if ind.inference_time > 0]
        
        return {
            'population_size': len(self.population),
            'generation': self.generation,
            'best_fitness': self.best_fitness,
            'avg_fitness': sum(fitnesses) / len(fitnesses) if fitnesses else float('inf'),
            'best_depth': self.best_individual.depth if self.best_individual else 0,
            'avg_depth': sum(depths) / len(depths) if depths else 0,
            'avg_nodes': sum(node_counts) / len(node_counts) if node_counts else 0,
            'avg_time_ms': sum(times) / len(times) if times else 0,
        }
    
    def print_stats(self) -> None:
        """Print population statistics."""
        stats = self.get_stats()
        print("=" * 50)
        print(f"Generation:        {stats['generation']}")
        print(f"Population Size:   {stats['population_size']}")
        print(f"Best Fitness:      {stats['best_fitness']:.6f}")
        print(f"Avg Fitness:       {stats['avg_fitness']:.6f}")
        print(f"Best Depth:        {stats['best_depth']}")
        print(f"Avg Depth:         {stats['avg_depth']:.2f}")
        print(f"Avg Nodes:         {stats['avg_nodes']:.2f}")
        print(f"Avg Time:          {stats['avg_time_ms']:.3f} ms")
        if self.best_individual:
            print(f"Best Expression:   {self.best_individual.get_expression()}")
        print("=" * 50)