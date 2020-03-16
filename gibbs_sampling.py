import numpy as np
from scipy.special import betaln

# All'inizio solo un componente, poi si assegna il nodo n in base alla probability of belonging to the first component o a una nuova

# example inputs
X = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])  # adjacency matrix
a, b, A = 1, 2, 10  # hyperparameters
T = 1

# initialization
N = X.shape[0]  # number of nodes in the network
z = np.ones((N, 1))  # cluster of each node (at the beginning all the nodes belong to the same node)
Z = []

# inside loop
for t in range(T):
    for n in range(N):
        nn = [x for x in range(N) if x != n]  # for first iteration (indices not considered)
        K = z.shape[1]  # number of components
        m = sum(z[nn, :])[:, np.newaxis]  # number of nodes for each components
        M = np.tile(m, (1, K)) 

        # X[nn, nn] in matlab is X_nn_nn
        X_nn_nn = np.delete(X, n, 0)
        X_nn_nn = np.delete(X_nn_nn, n, 1)
        M1 = z[nn, :].T@X_nn_nn@z[nn, :] - np.diag(sum(X_nn_nn@z[nn, :]*z[nn, :])/2)  # number of links between components
        M0 = m*m.T - np.diag(m*(m+1) / 2) - M1  # number of no-links between components
        r = z[nn, :].T@X[nn, n][:, np.newaxis]  # number of links from node n(0)
        R = np.tile(r, (1, K))
        beta1 = betaln(M1+R+a, M0+M-R+b)-betaln(M1+a, M0+b)
        beta1 = beta1.ravel()
        beta2 = betaln(r+a, m-r+b)-betaln(a, b)
        beta2 = beta2.ravel()
        beta_arr = np.concatenate((beta1, beta2))
        logP = sum(beta_arr, 1) + np.log(np.append(m, A))
        P = np.exp(logP-max(logP))
        rand_arr = np.random.rand() < np.cumsum(P)/sum(P)
        i = rand_arr.tolist().index(True)
        z[n, :] = 0

        try:
            z[n, i] = 1  # aggiungere se outofbounds
        except IndexError:
            new_arr = np.zeros((z.shape[0], i - z.shape[1] + 1))
            z = np.concatenate((z, new_arr), axis=1)
            z[n, i] = 1

        # remove empty component
        empty_cluster = np.argwhere(np.sum(z, axis=0) == 0).squeeze()
        z = np.delete(z, empty_cluster, 1)
    Z.append(z)


m = np.array([[1],[2]])
np.tile(m, (1, 2))
m = np.array([1,2])
