import numpy as np
import pandas as pd
from gibbs_sampling import irm, get_partitions
from itertools import islice


if __name__ == "__main__":
    adj = pd.read_csv("./data/output/adjmat_leg9.csv").drop(columns=['Unnamed: 0'])
    X = adj.values
    Z = irm(X=X, A=1, T=10)
    partitions = get_partitions(Z, burn_in_factor=1)
    first_ten = list(islice(partitions.items(), 2))
    print(first_ten)
