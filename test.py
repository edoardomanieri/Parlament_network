import numpy as np
from scipy.special import betaln
from irm import beta_bernoulli, crp_parameter
from network import get_partitions, co_clustering_matrix
from itertools import islice

if __name__ == "__main__":

    # example inputs
    X = np.array([[1, 1, 0, 0], [1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])                      # adjacency matrix
    Z = beta_bernoulli(X=X, A=1, T=5)
    partitions = get_partitions(Z, burn_in_factor=1)
    first_ten = dict(islice(partitions.items(), 5))
    print(first_ten)
    crp_parameter(650, 10)
    z1 = np.array([[1],[1],[1],[1]])
    z2 = np.array([[1,0], [1,0],[0,1],[0,1]])
    Z = [z1,z2]
    co_clustering_matrix(Z, burn_in_factor=1)

