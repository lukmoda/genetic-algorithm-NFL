import numpy as np
from random import random
from tqdm import tqdm
from .individual import Individual

class GA():
    """
    Genetic Algorithm class, instantiate and call <fit> method to build entire pipeline.
    """
    def __init__(self, pop_size):
        """
        Receive number of Individuals of the population.
        Best solution initialized as 0.
        """
        self.pop_size = pop_size
        self.population = []
        self.generation = 0
        self.best_solution = 0
        self.solutions_list = []
        
    def initialize_population(self, n_players, roster_space, cap_limit, strategy, positions, ratings, salaries):
        """
        Initialize population, appending Individuals until <pop_size> is reached.
        """
        for _ in range(self.pop_size):
            self.population.append(Individual(n_players, roster_space, cap_limit, strategy, positions, ratings, salaries))
        #update best solution to be first Individual
        self.best_solution = self.population[0]

    def sort_population(self):
        """
        Sort population by grade in ascending order.
        """
        self.population = sorted(self.population,
                                key = lambda population: population.grade,
                                reverse = True)
        
    def best_individual(self, individual):
        """
        Helper method that updates best solution if Individual grade is better 
        than previous best solution.
        """
        if individual.grade > self.best_solution.grade:
            self.best_solution = individual
            
    def sum_grades(self):
        """
        Helper method to sum grades from all Individuals of the population.
        """
        return sum([individual.grade for individual in self.population])
    
    def select_parent(self, sum_grades):
        """
        Select parent using Roulette Wheel Selection.
        """
        parent = -1
        random_val = random() * sum_grades
        tt = 0
        i = 0
        while i < len(self.population) and tt < random_val:
            tt += self.population[i].grade
            parent += 1
            i += 1
        return parent
    
    def visualize_generation(self):
        """
        Helper method to print results by generation.
        Prints generation, best grade and cap used by best solution.
        """
        #since population is sorted, first Individual is the best
        best = self.population[0]
        print('G: {}\nGrade: {}\nCap Used: {}'.format(best.generation, best.grade, best.cap_used))

    def fit(self, mutation_rate, mutation_power, num_generations, n_players, roster_space, cap_limit, strategy, positions, ratings, salaries, verbose=True):
        """
        Fit Genectic Algorithm, returning the chromosome from the best solution.

        Receive mutation rate, mutation power, number of generations, number of players
        on the database, roster space, cap limit, strategy used, positions, ratings and
        salaries arrays and a verbose parameter that regulates if each generation
        is printed.
        """
        self.initialize_population(n_players, roster_space, cap_limit, strategy, positions, ratings, salaries)
        for individual in self.population:
            individual.evaluation()
        
        self.sort_population()
        self.best_solution = self.population[0]
        self.solutions_list.append(self.best_solution.grade)
        if verbose:
            self.visualize_generation()
        
        for _ in tqdm(range(num_generations)):
            grade_sum = self.sum_grades()
            new_population = []
            
            for _ in range(0, self.pop_size, 2):
                parent1 = self.select_parent(grade_sum)
                parent2 = self.select_parent(grade_sum)
                
                children = self.population[parent1].crossover(self.population[parent2])
                
                new_population.append(children[0].mutation(mutation_rate, mutation_power))
                new_population.append(children[1].mutation(mutation_rate, mutation_power))
            
            self.population = list(new_population)
            for individual in self.population:
                individual.evaluation()
            self.sort_population()
            if verbose:
                self.visualize_generation()
            best = self.population[0]
            self.solutions_list.append(best.grade)
            self.best_individual(best)
        
        print('\n\nBest Solution: \nG: {}\nGrade: {}\nCap Used: {}'.format(self.best_solution.generation, self.best_solution.grade, self.best_solution.cap_used))
        
        return self.best_solution.chromosome