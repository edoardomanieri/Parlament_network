import numpy as np
from scipy.special import betaln
from irm import beta_bernoulli
from network import get_partitions
from itertools import islice

if __name__ == "__main__":

    # example inputs
    X = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])                      # adjacency matrix
    Z = beta_bernoulli(X=X, A=1, T=5)
    partitions = get_partitions(Z, burn_in_factor=1)
    first_ten = dict(islice(partitions.items(), 5))
    print(first_ten)