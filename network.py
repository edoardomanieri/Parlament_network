import numpy as np
import pandas as pd
from gibbs_sampling import irm, get_partitions

adj = pd.read_csv("./data/output/adjmat_leg9.csv").drop(columns = ['Unnamed: 0'])
X = adj.values
Z = irm(X, T=1)  # 13s for each sweep -> 37h for 10000 sweeps
get_partitions(Z)