import numpy as np
import csv
from tqdm import tqdm
from classes.ga import GA
from functions import preproc, get_players, save_results
from conf import *
import time
    
df, positions, ratings, salaries, N_PLAYERS = preproc()
best_grades = []

for i in tqdm(range(100)):
    ga = GA(POPULATION_SIZE)
    result = ga.fit(MUTATION_RATE, MUTATION_POWER, NUM_GENERATIONS, N_PLAYERS, ROSTER_SPACE, CAP_SPACE, STRATEGY, positions, ratings, salaries, verbose=False)
    best_grades.append(np.max(ga.solutions_list))
    df_result = get_players(df, result)
    save_results(df_result, '{}_{}.csv'.format(STRATEGY, i+1), 
                ga.solutions_list, '{}_{}.png'.format(STRATEGY, i+1))

with open('output/{}_grades.csv'.format(STRATEGY), 'w') as f:
    wr = csv.writer(f, quoting=csv.QUOTE_ALL)
    [wr.writerow([g]) for g in best_grades]
    