import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

font = {'weight' : 'bold',
        'size'   : 14}
plt.rc('font', **font)

os.chdir('output/')
files = glob.glob('*.csv')
grades = [g for g in files if 'grade' in g]
results = [g for g in files if 'grade' not in g]
balanced = [f for f in results if 'Balanced' in f]
elite = [f for f in results if 'Elite' in f]
playmakers = [f for f in results if 'Playmakers' in f]
defense = [f for f in results if 'Defense' in f]
trenches = [f for f in results if 'Trenches' in f]
starters = [f for f in results if 'Starters' in f]

starters.sort(key= lambda x: float(x.strip('Starters_.csv')))
balanced.sort(key= lambda x: float(x.strip('Balanced_.csv')))
elite.sort(key= lambda x: float(x.strip('Elite QB_.csv')))
playmakers.sort(key= lambda x: float(x.strip('Playmakers_.csv')))
defense.sort(key= lambda x: float(x.strip('Defense_.csv')))
trenches.sort(key= lambda x: float(x.strip('Trenches_.csv')))
starters.sort(key= lambda x: float(x.strip('Starters_.csv')))

df_full = pd.concat((pd.read_csv(f) for f in results), ignore_index=True)
df_bal = pd.concat((pd.read_csv(f) for f in balanced), ignore_index=True)
df_elite = pd.concat((pd.read_csv(f) for f in elite), ignore_index=True)
df_play = pd.concat((pd.read_csv(f) for f in playmakers), ignore_index=True)
df_def = pd.concat((pd.read_csv(f) for f in defense), ignore_index=True)
df_tren = pd.concat((pd.read_csv(f) for f in trenches), ignore_index=True)
df_start = pd.concat((pd.read_csv(f) for f in starters), ignore_index=True)

# Top 10 Teams
x = df_start.team.value_counts()[0:10].index
y = df_start.team.value_counts()[0:10]
plt.figure(figsize=(16,12))
line = plt.bar(x, y)
plt.xlabel('Team')
plt.ylabel("Players selected")
plt.title("Top 10 Teams - Starters")
plt.xticks(rotation=45)
for i in range(10):
    plt.annotate(str(y[i]), xy=(x[i], y[i]), ha='center', va='bottom')
plt.show()

# Bottom 10 Teams
x = df_full.team.value_counts(ascending=True)[0:10].index
y = df_full.team.value_counts(ascending=True)[0:10]
plt.figure(figsize=(16,12))
line = plt.bar(x, y)
plt.xlabel('Team')
plt.ylabel("Players selected")
plt.title("Bottom 10 Teams - General")
plt.xticks(rotation=45)
for i in range(10):
    plt.annotate(str(y[i]), xy=(x[i], y[i]), ha='center', va='bottom')
plt.show()

# Top 30 Players
top_players = df_tren['Name'].str.replace('\s+', '_').value_counts()[:30]
players = [pl+' ' for pl in top_players.index]
values = [pl for pl in top_players]
text_list = []
for p,v in zip(players, values):
    text_list.append(p*v)
text = ''.join(text_list)

wordcloud = WordCloud(max_font_size=50, max_words=100, collocations=False, background_color="white", colormap="winter").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Top 10 starting QBs
qbs = []
for file in balanced:
    dd = pd.read_csv(file)
    dd.loc[dd['position'] == 'QB', 'Name'] = dd.loc[dd['position'] == 'QB', 'Name'].str.replace('\s+', '_')
    qbs.append(dd[dd['position'] == 'QB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'Name'])

text = ' '.join(qbs)

wordcloud = WordCloud(max_font_size=50, max_words=100, collocations=False, background_color="white", colormap="winter").generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Calculate Overalls
ovr_gen = []
ovr_def = []
ovr_off = []
ovr_start = []
ages = []

for file in trenches:
    dd = pd.read_csv('output/{}'.format(file))    
    rat_def = []
    rat_off = []
    rat_def.extend([val for val in (dd[dd['position'] == 'DL'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:3, 'overall_rating'])])
    rat_def.extend([val for val in (dd[dd['position'] == 'LB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:2, 'overall_rating'])])
    rat_def.extend([val for val in (dd[dd['position'] == 'DB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:3, 'overall_rating'])])
    
    rat_off.append(dd[dd['position'] == 'QB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'overall_rating'])
    rat_off.extend([val for val in (dd[dd['position'] == 'HB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:1, 'overall_rating'])])
    rat_off.extend([val for val in (dd[dd['position'] == 'WR'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:3, 'overall_rating'])])
    rat_off.append(dd[dd['position'] == 'TE'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'overall_rating'])
    rat_off.append(dd[dd['position'] == 'C'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'overall_rating'])
    rat_off.extend([val for val in (dd[dd['position'] == 'G'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:1, 'overall_rating'])])
    rat_off.extend([val for val in (dd[dd['position'] == 'T'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:1, 'overall_rating'])])
    
    ovr_gen.append(dd['overall_rating'].mean())
    ovr_def.append(np.mean(rat_def))
    ovr_off.append(np.mean(rat_off))
    ovr_start.append(np.mean(np.concatenate([rat_def, rat_off])))
    ages.append(dd['age'].mean())
    
ovr_strat = pd.read_csv('output/Trenches_grades.csv', header=None).values.reshape(-1)
d = {'Ovr_Strategy': ovr_strat, 'Ovr_Starters': ovr_start, 'Ovr_Balanced': ovr_gen, 
     'Ovr_Offense': ovr_off, 'Ovr_Defense': ovr_def, 'Avg_Age': ages}

pd.DataFrame(d).to_csv('analysis/overall_trenches.csv', index=False)

# Compare Overalls
df_ovr = pd.read_csv('analysis/overall_defense.csv')

corr = df_ovr.corr()
plt.figure(figsize=(12,8))
plt.title('Correlation Heatmap - Starters')
cmap = sns.color_palette('RdBu_r')
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap=cmap, annot=True)
plt.show()

#pd.plotting.scatter_matrix(df_ovr.iloc[:, :-1], diagonal='kde', figsize=(24,24))

plt.figure(figsize=(24,18))
plt.title('Overalls Comparison - Starters')
plt.xlabel('Solution')
plt.ylabel('Overall')
plt.ylim([72, 89])
plt.xticks(np.arange(0, 101, 5))
plt.yticks(np.arange(72, 90, 1))
plt.scatter(df_ovr.index, df_ovr['Ovr_Strategy'], color='green', marker='o', label='Strategy')
plt.plot(df_ovr['Ovr_Strategy'], color='green')
plt.scatter(df_ovr.index, df_ovr['Ovr_Defense'], color='blue', marker='*', label='Defense')
plt.plot(df_ovr['Ovr_Defense'], color='blue')
plt.scatter(df_ovr.index, df_ovr['Ovr_Offense'], color='red', marker='^', label='Offense')
plt.plot(df_ovr['Ovr_Offense'], color='red')
plt.scatter(df_ovr.index, df_ovr['Ovr_Starters'], color='black', marker='P', label='Starters')
plt.plot(df_ovr['Ovr_Starters'], color='black')
plt.scatter(df_ovr.index, df_ovr['Ovr_Balanced'], color='cyan', marker='D', label='Balanced')
plt.plot(df_ovr['Ovr_Balanced'], color='cyan')
plt.legend(loc='best')
plt.show()

# Print Starters

def print_starters(file):
    dd = pd.read_csv(file)
    dd['Name'] = dd['Name'].str.title()
    
    st_dl = dd[dd['position'] == 'DL'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:3, 'Name'].tolist()
    st_dl_rt = dd[dd['position'] == 'DL'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:3, 'overall_rating'].tolist()
    bk_dl = dd[dd['position'] == 'DL'].sort_values(by='overall_rating', ascending=False).reset_index().loc[4:5, 'Name'].tolist()
    bk_dl_rt = dd[dd['position'] == 'DL'].sort_values(by='overall_rating', ascending=False).reset_index().loc[4:5, 'overall_rating'].tolist()
    
    st_lb = dd[dd['position'] == 'LB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:2, 'Name'].tolist()
    st_lb_rt = dd[dd['position'] == 'LB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:2, 'overall_rating'].tolist()
    bk_lb = dd[dd['position'] == 'LB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[3:4, 'Name'].tolist()
    bk_lb_rt = dd[dd['position'] == 'LB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[3:4, 'overall_rating'].tolist()
    
    st_db = dd[dd['position'] == 'DB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:3, 'Name'].tolist()
    st_db_rt = dd[dd['position'] == 'DB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:3, 'overall_rating'].tolist()
    bk_db = dd[dd['position'] == 'DB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[4:6, 'Name'].tolist()
    bk_db_rt = dd[dd['position'] == 'DB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[4:6, 'overall_rating'].tolist()
    
    st_qb = dd[dd['position'] == 'QB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'Name']
    st_qb_rt = dd[dd['position'] == 'QB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'overall_rating']
    
    bk_qb = dd[dd['position'] == 'QB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[1, 'Name']
    bk_qb_rt = dd[dd['position'] == 'QB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[1, 'overall_rating']
    
    st_c = dd[dd['position'] == 'C'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'Name']
    st_c_rt = dd[dd['position'] == 'C'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'overall_rating']
    
    st_hb = dd[dd['position'] == 'HB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'Name']
    st_hb_rt = dd[dd['position'] == 'HB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'overall_rating']
    bk_hb = dd[dd['position'] == 'HB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[1:2, 'Name'].tolist()
    bk_hb_rt = dd[dd['position'] == 'HB'].sort_values(by='overall_rating', ascending=False).reset_index().loc[1:2, 'overall_rating'].tolist()
    
    st_te = dd[dd['position'] == 'TE'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'Name']
    st_te_rt = dd[dd['position'] == 'TE'].sort_values(by='overall_rating', ascending=False).reset_index().loc[0, 'overall_rating']
    bk_te = dd[dd['position'] == 'TE'].sort_values(by='overall_rating', ascending=False).reset_index().loc[1, 'Name']
    bk_te_rt = dd[dd['position'] == 'TE'].sort_values(by='overall_rating', ascending=False).reset_index().loc[1, 'overall_rating']
    
    st_g = dd[dd['position'] == 'G'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:1, 'Name'].tolist()
    st_g_rt = dd[dd['position'] == 'G'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:1, 'overall_rating'].tolist()
    bk_g = dd[dd['position'] == 'G'].sort_values(by='overall_rating', ascending=False).reset_index().loc[2, 'Name']
    bk_g_rt = dd[dd['position'] == 'G'].sort_values(by='overall_rating', ascending=False).reset_index().loc[2, 'overall_rating']
    
    st_t = dd[dd['position'] == 'T'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:1, 'Name'].tolist()
    st_t_rt = dd[dd['position'] == 'T'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:1, 'overall_rating'].tolist()
    bk_t = dd[dd['position'] == 'T'].sort_values(by='overall_rating', ascending=False).reset_index().loc[2, 'Name']
    bk_t_rt = dd[dd['position'] == 'T'].sort_values(by='overall_rating', ascending=False).reset_index().loc[2, 'overall_rating']
    
    st_wr = dd[dd['position'] == 'WR'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:2, 'Name'].tolist()
    st_wr_rt = dd[dd['position'] == 'WR'].sort_values(by='overall_rating', ascending=False).reset_index().loc[:2, 'overall_rating'].tolist()
    bk_wr = dd[dd['position'] == 'WR'].sort_values(by='overall_rating', ascending=False).reset_index().loc[3:4, 'Name'].tolist()
    bk_wr_rt = dd[dd['position'] == 'WR'].sort_values(by='overall_rating', ascending=False).reset_index().loc[3:4, 'overall_rating'].tolist()
    
    others = dd[dd['position'].isin(['K', 'P', 'FB'])].sort_values(by='overall_rating', ascending=False).reset_index().loc[:3, 'Name'].tolist()
    others_rt = dd[dd['position'].isin(['K', 'P', 'FB'])].sort_values(by='overall_rating', ascending=False).reset_index().loc[:3, 'overall_rating'].tolist()
           
    print('Offense: ')
    print('QB {} ({})'.format(st_qb, st_qb_rt))
    print('Playmakers: HB {} ({}), TE {} ({}), WR {} ({}), WR {} ({}), WR {} ({})'.format(st_hb, st_hb_rt, st_te, st_te_rt,
          st_wr[0], st_wr_rt[0], st_wr[1], st_wr_rt[1], st_wr[2],st_wr_rt[2]))
    print('OL: C {} ({}), G {} ({}), G {} ({}), T {} ({}), T {} ({})'.format(st_c, st_c_rt, st_g[0],
          st_g_rt[0], st_g[1], st_g_rt[1], st_t[0], st_t_rt[0], st_t[1], st_t_rt[1]))
    print('\nDefense: ')
    print('DL: {} ({}), {} ({}), {} ({}), {} ({})'.format(st_dl[0], st_dl_rt[0], st_dl[1],
          st_dl_rt[1], st_dl[2], st_dl_rt[2], st_dl[3], st_dl_rt[3]))
    print('LB: {} ({}), {} ({}), {} ({})'.format(st_lb[0], st_lb_rt[0], st_lb[1], st_lb_rt[1],
          st_lb[2],st_lb_rt[2]))
    print('DB: {} ({}), {} ({}), {} ({}), {} ({})'.format(st_db[0], st_db_rt[0], st_db[1],
          st_db_rt[1], st_db[2], st_db_rt[2], st_db[3],st_db_rt[3]))
    print('''\nBackups: QB {} ({}), HB {} ({}), HB {} ({}), TE {} ({}), WR {} ({}), WR {} ({}), 
          G {} ({}), T {} ({}), DL {} ({}), DL {} ({}), LB {} ({}), LB {} ({}), DB {} ({}), 
          DB {} ({}), DB {} ({})'''.format(bk_qb, bk_qb_rt, bk_hb[0], bk_hb_rt[0], bk_hb[1], 
          bk_hb_rt[1], bk_te, bk_te_rt, bk_wr[0], bk_wr_rt[0], bk_wr[1], bk_wr_rt[1], 
          bk_g, bk_g_rt, bk_t, bk_t_rt, bk_dl[0], bk_dl_rt[0], bk_dl[1], bk_dl_rt[1], bk_lb[0],
          bk_lb_rt[0], bk_lb[1], bk_lb_rt[1], bk_db[0], bk_db_rt[0], bk_db[1], 
          bk_db_rt[1], bk_db[2], bk_db_rt[2]))
    
    if len(others) > 0:
        ot = list(np.array([[d, c] for d, c in zip(others, others_rt)]).reshape(-1))
        for index, item in enumerate(ot): 
            if index % 2 == 1: 
                ot[index] = '({})'.format(ot[index])
        print('\nOthers: {}'.format(' '.join(ot)))
    
print_starters('output/Starters_82.csv')