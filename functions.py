import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from classes.player import Player

def preproc():
    """
    Read Ratings and Contracts csvs and do all the preprocessing needed.

    Return cleansed DataFrame, positions, ratings and salaries arrays
    and number of players in the database.
    """
    df_players = pd.read_csv('input/M21_Ratings.csv')[['Name', 'position', 'team', 'age', 
                            'overall_rating']]
    df_contracts = pd.read_csv('input/NFL_Contracts.csv')[['Player', 'Team', 'Avg/Year']]
    
    df_players['Name'] = df_players['Name'].str.lower()
    df_players['Name'] = df_players['Name'].str.replace(".", "")
    df_players['Name'] = df_players['Name'].str.replace("iii", "")
    df_players['Name'] = df_players['Name'].str.replace("ii", "")
    df_players['Name'] = df_players['Name'].str.replace("jr", "")
    df_players['Name'] = df_players['Name'].str.replace("sr", "")
    df_players['Name'] = df_players['Name'].str.strip()
    
    #group some positions
    df_players.loc[(df_players['position'] == 'RG') | 
            (df_players['position'] == 'LG'), 'position'] = 'G'
    df_players.loc[(df_players['position'] == 'RT') | 
            (df_players['position'] == 'LT'), 'position'] = 'T'
    df_players.loc[(df_players['position'] == 'DT') | 
            (df_players['position'] == 'RE') | (df_players['position'] == 'LE'),
            'position'] = 'DL'
    df_players.loc[(df_players['position'] == 'MLB') | 
            (df_players['position'] == 'ROLB') | (df_players['position'] == 'LOLB'),
            'position'] = 'LB'
    df_players.loc[(df_players['position'] == 'CB') | 
            (df_players['position'] == 'SS') | (df_players['position'] == 'FS'),
            'position'] = 'DB'
            
    df_contracts['Player'] = df_contracts['Player'].str.lower() 
    df_contracts['Player'] = df_contracts['Player'].str.replace(".", "")
    df_contracts['Player'] = df_contracts['Player'].str.replace("iii", "")
    df_contracts['Player'] = df_contracts['Player'].str.replace("ii", "")
    df_contracts['Player'] = df_contracts['Player'].str.replace("jr", "")
    df_contracts['Player'] = df_contracts['Player'].str.replace("sr", "")
    df_contracts['Player'] = df_contracts['Player'].str.strip()
    
    df_contracts['Avg/Year'] = df_contracts['Avg/Year'].str.replace("$", "")
    df_contracts['Avg/Year'] = df_contracts['Avg/Year'].str.replace(",", "")
    df_contracts['Avg/Year'] = df_contracts['Avg/Year'].astype(int)
    
    df = pd.merge(df_players, df_contracts, left_on=['Name', 'team'], 
                  right_on=['Player', 'Team'], how='inner').drop(['Player', 'Team'], 
                           axis=1)
    
    #shuffle DataFrame, since original df is sorted by overall rating
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    players_list = []
    for i in range(len(df)):
        players_list.append(Player(df.loc[i, 'Name'], df.loc[i, 'position'],
                                   df.loc[i, 'age'], df.loc[i, 'overall_rating'],
                                   df.loc[i, 'Avg/Year']))
    positions = []
    ratings = []
    salaries = []    
    for player in players_list:
        positions.append(player.position)
        ratings.append(player.rating)
        salaries.append(player.salary)
    return df, positions, ratings, salaries, len(df)

def overall_strategy(individual, idxs):
    """
    Calculate evaluation metric based on strategy chosen.

    Receive indexes of players chosen by the solution and individual object.
    Metrics are calculated as follows:
    - Balanced: Simple average across all 53 players;
    - Elite QB: Best QB on roster has weight 10;
    - Playmakers: Best QB has weight 5 and best HB, TE and top 3 WR 
                    have weight 8;
    - Defense: Top 4 DL, top 3 LB and top 4 DB have weight 5;
    - Trenches: Top 2 G, top 2 T, top 4 DL and best C have weight 5;
    - Starters: Best HB, TE, C and top 2 G, top 2 T, top 3 WR, top 4 DL, 
                    top 3 LB and top 4 DB have weight 5. Best QB has weight 8.
    """
    if individual.strategy == 'Balanced':
        return np.mean([individual.ratings[i] for i in idxs])
    elif individual.strategy == 'Elite QB':
        idxs_qbs = [i for i in idxs if individual.positions[i] == 'QB']
        ind_qb_max = idxs_qbs[np.argmax([individual.ratings[i] for i in idxs_qbs])]
        weights = [1] * len(individual.ratings)
        weights[ind_qb_max] = 10
        return np.average([individual.ratings[i] for i in idxs], weights=[weights[i] for i in idxs])
    elif individual.strategy == 'Playmakers':
        idxs_qbs = [i for i in idxs if individual.positions[i] == 'QB']
        idxs_hbs = [i for i in idxs if individual.positions[i] == 'HB']
        idxs_tes = [i for i in idxs if individual.positions[i] == 'TE']
        idxs_wrs = [i for i in idxs if individual.positions[i] == 'WR']
        ind_qb_max = idxs_qbs[np.argmax([individual.ratings[i] for i in idxs_qbs])]
        ind_hb_max = idxs_hbs[np.argmax([individual.ratings[i] for i in idxs_hbs])]
        ind_te_max = idxs_tes[np.argmax([individual.ratings[i] for i in idxs_tes])]
        idxs_top3wr = [idxs_wrs[i] for i in np.array([individual.ratings[i] for i in idxs_wrs]).argsort()[-3:][::-1]]
        weights = [1] * len(individual.ratings)
        weights[ind_qb_max] = 5
        weights[ind_hb_max] = 8
        weights[ind_te_max] = 8
        for i in idxs_top3wr:
            weights[i] = 8
        return np.average([individual.ratings[i] for i in idxs], weights=[weights[i] for i in idxs])
    elif individual.strategy == 'Defense':
        idxs_dls = [i for i in idxs if individual.positions[i] == 'DL']
        idxs_lbs = [i for i in idxs if individual.positions[i] == 'LB']
        idxs_dbs = [i for i in idxs if individual.positions[i] == 'DB']
        idxs_top4dl = [idxs_dls[i] for i in np.array([individual.ratings[i] for i in idxs_dls]).argsort()[-4:][::-1]]
        idxs_top3lb = [idxs_lbs[i] for i in np.array([individual.ratings[i] for i in idxs_lbs]).argsort()[-3:][::-1]]
        idxs_top4db = [idxs_dbs[i] for i in np.array([individual.ratings[i] for i in idxs_dbs]).argsort()[-4:][::-1]]
        weights = [1] * len(individual.ratings)
        for i in np.concatenate([idxs_top4dl, idxs_top3lb, idxs_top4db]):
            weights[i] = 5
        return np.average([individual.ratings[i] for i in idxs], weights=[weights[i] for i in idxs])
    elif individual.strategy == 'Trenches':
        idxs_cs = [i for i in idxs if individual.positions[i] == 'C']
        idxs_gs = [i for i in idxs if individual.positions[i] == 'G']
        idxs_ts = [i for i in idxs if individual.positions[i] == 'T']
        idxs_dls = [i for i in idxs if individual.positions[i] == 'DL']
        idxs_top2g = [idxs_gs[i] for i in np.array([individual.ratings[i] for i in idxs_gs]).argsort()[-2:][::-1]]
        idxs_top2t = [idxs_ts[i] for i in np.array([individual.ratings[i] for i in idxs_ts]).argsort()[-2:][::-1]]
        idxs_top4dl = [idxs_dls[i] for i in np.array([individual.ratings[i] for i in idxs_dls]).argsort()[-4:][::-1]]
        ind_c_max = [idxs_cs[np.argmax([individual.ratings[i] for i in idxs_cs])]]
        weights = [1] * len(individual.ratings)
        for i in np.concatenate([idxs_top2g, idxs_top2t, idxs_top4dl, ind_c_max]):
            weights[i] = 5
        return np.average([individual.ratings[i] for i in idxs], weights=[weights[i] for i in idxs])
    elif individual.strategy == 'Starters':
        idxs_qbs = [i for i in idxs if individual.positions[i] == 'QB']            
        idxs_hbs = [i for i in idxs if individual.positions[i] == 'HB']
        idxs_tes = [i for i in idxs if individual.positions[i] == 'TE']
        idxs_wrs = [i for i in idxs if individual.positions[i] == 'WR']
        idxs_cs = [i for i in idxs if individual.positions[i] == 'C']
        idxs_gs = [i for i in idxs if individual.positions[i] == 'G']
        idxs_ts = [i for i in idxs if individual.positions[i] == 'T']
        idxs_dls = [i for i in idxs if individual.positions[i] == 'DL']
        idxs_lbs = [i for i in idxs if individual.positions[i] == 'LB']
        idxs_dbs = [i for i in idxs if individual.positions[i] == 'DB']
        ind_qb_max = idxs_qbs[np.argmax([individual.ratings[i] for i in idxs_qbs])]
        ind_hb_max = [idxs_hbs[np.argmax([individual.ratings[i] for i in idxs_hbs])]]
        ind_te_max = [idxs_tes[np.argmax([individual.ratings[i] for i in idxs_tes])]]
        idxs_top3wr = [idxs_wrs[i] for i in np.array([individual.ratings[i] for i in idxs_wrs]).argsort()[-3:][::-1]]    
        ind_c_max = [idxs_cs[np.argmax([individual.ratings[i] for i in idxs_cs])]]
        idxs_top2g = [idxs_gs[i] for i in np.array([individual.ratings[i] for i in idxs_gs]).argsort()[-2:][::-1]]
        idxs_top2t = [idxs_ts[i] for i in np.array([individual.ratings[i] for i in idxs_ts]).argsort()[-2:][::-1]]
        idxs_top4dl = [idxs_dls[i] for i in np.array([individual.ratings[i] for i in idxs_dls]).argsort()[-4:][::-1]]
        idxs_top3lb = [idxs_lbs[i] for i in np.array([individual.ratings[i] for i in idxs_lbs]).argsort()[-3:][::-1]]
        idxs_top4db = [idxs_dbs[i] for i in np.array([individual.ratings[i] for i in idxs_dbs]).argsort()[-4:][::-1]]
        weights = [1] * len(individual.ratings)
        weights[ind_qb_max] = 8
        for i in np.concatenate([ind_hb_max, ind_te_max, idxs_top3wr, ind_c_max,
                        idxs_top2g, idxs_top2t, idxs_top4dl, idxs_top3lb, idxs_top4db]):
            weights[i] = 5
        return np.average([individual.ratings[i] for i in idxs], weights=[weights[i] for i in idxs])

def validate_positions(individual):
    """
    Check if the individual chromosome respects expected positions in 
    a typical NFL 53 man roster.

    Return a boolean: True if 53 players chosen are NLF-Roster like, 
                        and False otherwise.
    The roster must have 2 or 3 QB, 2 or 3 C, between 5 and 8 WR, 2 and 5 TE,
                3 and 5 HB, 3 and 5 G, 3 and 5 T, 6 and 9 DL, 6 and 10 LB,
                7 and 11 DB and no more than 1 K and 1 P. Also, the backup
                QB can't have more than 77 overall, the backup C can't 
                have more than 75 overall and the 3rd HB can't have more 
                than 78 overall.
    """
    players_idxs = [idx for idx, val in enumerate(individual.chromosome) if val == 1]
    positions_list = [individual.positions[i] for i in players_idxs]
    freq = Counter(positions_list)
    if freq['QB'] > 1:
        idxs_qbs = [i for i in players_idxs if individual.positions[i] == 'QB']            
        idx_top2qb = np.array([individual.ratings[i] for i in idxs_qbs]).argsort()[-2:][::-1]
        if individual.ratings[idxs_qbs[idx_top2qb[1]]] > 77:
            return False
    if freq['C'] > 1:
        idxs_cs = [i for i in players_idxs if individual.positions[i] == 'C']            
        idx_top2c = np.array([individual.ratings[i] for i in idxs_cs]).argsort()[-2:][::-1]
        if individual.ratings[idxs_cs[idx_top2c[1]]] > 75:
            return False
    if freq['HB'] > 2:
        idxs_hbs = [i for i in players_idxs if individual.positions[i] == 'HB']            
        idx_top3hb = np.array([individual.ratings[i] for i in idxs_hbs]).argsort()[-3:][::-1]
        if individual.ratings[idxs_hbs[idx_top3hb[2]]] > 78:
            return False
    if (freq['QB'] < 2 or freq['QB'] > 3) or (freq['WR'] < 5 or freq['WR'] > 8) or (freq['TE'] < 2 or freq['TE'] > 5) or (freq['HB'] < 3 or freq['HB'] > 5) or (freq['C'] < 2 or freq['C'] > 3) or (freq['G'] < 3 or freq['G'] > 5) or (freq['T'] < 3 or freq['T'] > 5) or (freq['DL'] < 6 or freq['DL'] > 9) or (freq['LB'] < 6 or freq['LB'] > 10) or (freq['DB'] < 7 or freq['DB'] > 11) or freq['P'] > 1 or freq['K'] > 1:
        return False
    return True

def get_players(df, chromosome):
    """
    Helper function to get players from DataFrame based on chromosome.

    Return 53 man-roster DataFrame sorted by overall rating.
    """
    players_idxs = [idx for idx, val in enumerate(chromosome) if val == 1]
    return df.loc[players_idxs].sort_values(by='overall_rating', ascending=False)
    
def save_results(df, csv_name, solutions_list, fig_name):
    """
    Save the resulting solution DataFrame in a csv and the plot of the evolution
    of best grade by generation in an image.
    """
    df.to_csv('output/{}'.format(csv_name), index=False)
    fig, ax = plt.subplots()
    ax.plot(solutions_list)
    ax.scatter(np.argmax(solutions_list), np.max(solutions_list), s=50, c='red',
               label='G: {}, {:.3f}'.format(np.argmax(solutions_list), np.max(solutions_list)))
    ax.set_xlabel('Generation')
    ax.set_ylabel('Average Overall')
    ax.set_title('Best Solution over Generations')
    ax.legend(loc='best')
    fig.savefig('output/{}'.format(fig_name))