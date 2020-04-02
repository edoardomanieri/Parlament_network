import pandas as pd
import numpy as np
from itertools import combinations

### bills
df_bills = pd.read_csv('data/bills-ca.csv')

# preprocessing
df_bills['cosponsors'] = df_bills['cosponsors'].str.split(';')
df_bills['authors'] = df_bills['authors'].str.split(';')
df_bills['date'] = pd.to_datetime(df_bills['date'], format="%Y%m%d")
df_bills['authors'] = df_bills['authors'].apply(lambda d: d if isinstance(d, list) else [])
df_bills['cosponsors'] = df_bills['cosponsors'].apply(lambda d: d if isinstance(d, list) else [])

# put together authors and cosponsors (baseline model)
df_bills['voters'] = df_bills['authors'] + df_bills['cosponsors']
df_bills = df_bills.explode('voters').reset_index()
df_bills.rename(columns={'index': 'law'}, inplace=True)
df_bills.drop(columns=['authors', 'cosponsors'], inplace=True)
df_bills.dropna(subset=['voters'], inplace=True)

# create adjacency matrix for each legislature
adjacency_matrices = []
for leg in df_bills['legislature'].unique():
    df_leg = df_bills.loc[df_bills['legislature'] == leg, ['voters', 'law']]
    adj = np.zeros((len(df_leg['voters'].unique()), len(df_leg['voters'].unique())))
    adj = np.fill_diagonal(adj, 1)
    df = pd.DataFrame(data=adj, index=df_leg['voters'].unique(), columns=df_leg['voters'].unique())
    dfg = df_leg.groupby('law')['voters'].apply(list).reset_index(name='new')
    for el in dfg['new']:
        for comb in combinations(el, 2):
            df.loc[comb[0], comb[1]] = 1
            df.loc[comb[1], comb[0]] = 1
    adjacency_matrices.append(df)

# save adjacency matrix
for i, leg in enumerate(df_bills['legislature'].unique()):
    adjacency_matrices[i].to_csv(f"./data/output/adjmat_leg{leg}.csv")


### sponsors
df_sponsors = pd.read_csv('data/sponsors-ca.csv')

# preprocessing
id_pos = df_sponsors['url'].str.rfind('/')[0] + 1
df_sponsors['id'] = df_sponsors['url'].str[id_pos:]
