import numpy as np
from random import random, sample
from functions import validate_positions, overall_strategy

class Individual():
    """
    Create an Individual for the Genectic Algorithm, with
    evaluation metric, crossover and mutation methods.
    """
    def __init__(self, n_players, roster_space, cap_limit, strategy, positions, ratings, salaries, generation=0):
        """
        Instantiate Individual with all variables needed for the simulation.
        Chromosome represents <roster_space> randomly chosen players from <n_players>.
        """
        self.n_players = n_players
        self.roster_space = roster_space
        self.cap_limit = cap_limit
        self.strategy = strategy
        self.positions = positions
        self.ratings = ratings
        self.salaries = salaries
        self.grade = 0
        self.cap_used = 0
        self.generation = generation
        
        arr = np.array([1] * roster_space + [0] * (n_players-roster_space))
        np.random.shuffle(arr)
        self.chromosome = list(arr)      
        
    def evaluation(self):
        """
        Update grade of Individual.

        If the sum of salaries is higher than the cap limit, or the positions do
        not pass the positions_check validation, grade is 1
        Otherwise, return the calculation based on the strategy, as specified
        in overall_strategy function.
        """
        idxs_one = [idx for idx, val in enumerate(self.chromosome) if val == 1]
        salaries_sum = sum([self.salaries[i] for i in idxs_one])
        positions_check = validate_positions(self)
        if salaries_sum > self.cap_limit or positions_check == False:
            grade = 1
        else:
            grade = overall_strategy(self, idxs_one)
        self.grade = grade
        self.cap_used = salaries_sum
        
    def crossover(self, partner):
        """
        Perform one-point crossover (reproduction) and return children.

        After crossover is applied, need to check if children have more or less
        than 53 players. If not, correct so solutions have 53 players.
        """
        #choose one index at random
        index_cut = round(random()  * len(self.chromosome))
        children = [Individual(self.n_players, self.roster_space, self.cap_limit, self.strategy, self.positions, self.ratings, self.salaries, self.generation + 1),
                  Individual(self.n_players, self.roster_space, self.cap_limit, self.strategy, self.positions, self.ratings, self.salaries, self.generation + 1)]
        #children receive parent's swapped genes
        children[0].chromosome = partner.chromosome[0:index_cut] + self.chromosome[index_cut::]
        children[1].chromosome = self.chromosome[0:index_cut] + partner.chromosome[index_cut::]
        #correct children with more or less than 53 players
        for i in range(len(children)):
            n_ones = sum(children[i].chromosome)
            if n_ones < 53:
                ones_to_fill = 53 - n_ones
                idxs_zero = [idx for idx, val in enumerate(children[i].chromosome) if val == 0]
                idxs_change = sample(idxs_zero, ones_to_fill)
                for idx in idxs_change:
                    children[i].chromosome[idx] = 1
            elif n_ones > 53:
                zeros_to_fill = n_ones - 53
                idxs_one = [idx for idx, val in enumerate(children[i].chromosome) if val == 1]
                idxs_change = sample(idxs_one, zeros_to_fill)
                for idx in idxs_change:
                    children[i].chromosome[idx] = 0
        return children
    
    def mutation(self, mutation_rate, mutation_power):
        """
        Perform mutation on Individual, if <mutation_rate> is bigger than random number.

        If mutation is activated, <mutation_power> genes that were 0 will turn to 1, and
        <mutation_power> genes that were 1 will turn to 0. This way, solutions still
        have 53 players.
        """
        if random() < mutation_rate:
            idxs_zero = [idx for idx, val in enumerate(self.chromosome) if val == 0]
            idxs_one = [idx for idx, val in enumerate(self.chromosome) if val == 1]
            idxs_new_zeros = sample(idxs_one, mutation_power)
            idxs_new_ones = sample(idxs_zero, mutation_power)
            for idx in idxs_new_zeros:
                self.chromosome[idx] = 0
            for idx in idxs_new_ones:
                self.chromosome[idx] = 1
        return self