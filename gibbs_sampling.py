import numpy as np
import pandas as pd
from scipy.special import betaln
from operator import itemgetter
from collections import defaultdict


# example inputs
X = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])                      # adjacency matrix


def irm(X, a=1, b=1, A=2,  T=100):

    # initialization
    N = X.shape[0]                                                                          # number of nodes in the network
    Z = []

    for _ in range(T):
        z = np.ones((N, 1))                                                                 # (not in matlab impl) 
        for n in range(N):                                                                  # component of each node (at the beginning all nodes belong to the same component)
            nn = [x for x in range(N) if x != n]                                            # for first iteration (indices not considered)
            K = z.shape[1]                                                                  # number of components
            m = sum(z[nn, :])[:, np.newaxis]                                                # number of nodes for each components
            M = np.tile(m, (1, K))

            # X[nn, nn] in matlab is X_nn_nn
            X_nn_nn = np.delete(X, n, 0)
            X_nn_nn = np.delete(X_nn_nn, n, 1)
            M1 = z[nn, :].T@X_nn_nn@z[nn, :] - np.diag(sum(X_nn_nn@z[nn, :]*z[nn, :])/2)    # number of links between components
            M0 = m@m.T - np.diag((m*(m+1)).ravel() / 2) - M1                                # number of no-links between components
            r = z[nn, :].T@X[nn, n][:, np.newaxis]                                          # number of links from node n(0)
            R = np.tile(r, (1, K))
            beta_old_comps = betaln(M1+R+a, M0+M-R+b)-betaln(M1+a, M0+b)                    # one column for each component
            np.fill_diagonal(beta_old_comps, 0)                                             # not in Matlab implementation
            beta_new_comp = betaln(r+a, m-r+b)-betaln(a, b)                                 # new component's values
            likelihood_change = sum(np.concatenate((beta_old_comps, beta_new_comp), axis=1), 0)
            prior_change = np.log(np.append(m, A))
            logP = likelihood_change + prior_change                                         # Log prob of n belonging to existing or new component
            P = np.exp(logP-logP.max())                                                     # Convert from log probability
            rand_arr = np.random.rand() < np.cumsum(P)/P.sum()                              # Random component according to P
            i = rand_arr.tolist().index(True)
            z[n, :] = 0

            new_arr = np.zeros((z.shape[0], 1))
            z = np.concatenate((z, new_arr), axis=1)
            z[n, i] = 1

            empty_cluster = np.argwhere(np.sum(z, axis=0) == 0).squeeze()                   # remove empty component
            z = np.delete(z, empty_cluster, 1)
        Z.append(z)
    return Z


def get_partitions(Z):
    res = defaultdict(int)
    for z in Z:
        df = pd.DataFrame(data=np.argwhere(z), columns=['node', 'group'])
        df = df.groupby('group')['node'].apply(tuple)
        tmp = tuple(sorted(df.values.tolist(), key=itemgetter(0)))
        res[tmp] += 1
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


if __name__ == "__main__":
    Z = irm(X=X, T=50)
    print(get_partitions(Z))