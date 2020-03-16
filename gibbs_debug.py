import numpy as np
from scipy.special import betaln


# ALl'inizio solo un componente, poi si assegna il nodo n in base alla probability of belonging to the first component o a una nuova

# inputs
X = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]])  # adjacency matrix
a, b, A = 1, 2, 10  # hyperparameters

# initialization
N = X.shape[0]  # number of nodes in the network
z = np.ones((N, 1))  # custer of each node
#Z not needed for one iteration

# inside loop
nn = [1, 2, 3]  # for first iteration (indices not considered)
K = z.shape[1]  # number of components
m = sum(z[nn, :]).T  # number of nodes for each components
M = np.tile(m, (1, K))

# X[nn, nn] is a = np.delete(X, n, 0) a = np.delete(X, n, 1)
X_nn_nn = np.delete(X, 0, 0) 
X_nn_nn = np.delete(X_nn_nn, 0, 1)
M1 = z[nn, :].T@X_nn_nn@z[nn, :] - np.diag(sum(X_nn_nn@z[nn, :]*z[nn, :])/2)  # number of links between components
M0 = m*m.T - np.diag(m*(m+1) / 2) - M1  # number of no-links between components
r = z[nn, :].T@X[nn, 0]  # here 0 should be n, number of links from node n(0)
R = np.tile(r, (1, K))
beta1 = betaln(M1+R+a,M0+M-R+b)-betaln(M1+a,M0+b)
beta1 = beta1.ravel()
beta2 = betaln(r+a,m-r+b)-betaln(a,b)
beta_arr = np.concatenate((beta1, beta2))
logP = sum(beta_arr, 1) + np.log(np.append(m, A))
P = np.exp(logP-max(logP))
rand_arr = np.random.rand()<np.cumsum(P)/sum(P)
i = rand_arr.tolist().index(True)
z[0, :] = 0
try:
    z[0, i] = 1 #aggiungere se outofbounds
except IndexError:
    new_arr = np.zeros((z.shape[0], i - z.shape[1] + 1))
    z = np.concatenate((z, new_arr), axis=1)
    z[0, i] = 1