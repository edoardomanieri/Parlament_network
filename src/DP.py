from scipy.special import digamma, polygamma, betaln, gamma
import numpy as np
from math import exp


def f(theta, n, p):
    return theta*(digamma(theta + n) - digamma(theta)) - p


def df(theta, n):
    return digamma(theta + n) + theta*polygamma(1, theta + n) - theta*polygamma(1, theta) - digamma(theta)


def dx(f, theta, n, p):
    return abs(0-f(theta, n, p))


def newtons_method(f, df, theta0, n, p, e, max_iter):
    delta = dx(f, theta0, n, p)
    i = 0
    while delta > e and i < max_iter:
        theta0 = theta0 - f(theta0, n, p)/df(theta0, n)
        delta = dx(f, theta0, n, p)
        i += 1
    if i >= max_iter:
        print("reached max_iter, consider increasing")
    return theta0, f(theta0, n, p)


def crp_parameters(n_politicians, n_parties, max_iter):
    e = 1e-10
    max_iter = 100000
    x0 = 1
    n = n_politicians
    p = n_parties
    param, value = newtons_method(f, df, x0, n, p, e, max_iter)
    print(f"CRP parameter: {param}")
    print(f"value of the function: {value}")
    return param


def beta_bernoulli_irm(X, a=1, b=1, theta=6,  T=100):

    # number of nodes in the network
    N = X.shape[0]
    z = np.ones((N, 1))
    Z = []
    np.fill_diagonal(X, 0)
    eps = 1e-200

    for _ in range(T):
        # Component of each node (at the beginning all nodes belong to the same component)
        for n in range(N):
            # indices considered
            nn = [x for x in range(N) if x != n]
            # number of components
            K = z.shape[1]
            # number of nodes for each components
            m = sum(z[nn, :])[:, np.newaxis]
            M = np.tile(m, (1, K))

            # X[nn, nn] in matlab is X_nn_nn
            X_nn_nn = np.delete(X, n, 0)
            X_nn_nn = np.delete(X_nn_nn, n, 1)
            # number of links between components
            M1 = z[nn, :].T@X_nn_nn@z[nn, :] - np.diag(sum(X_nn_nn@z[nn, :]*z[nn, :])/2)
            # number of no-links between components
            M0 = m@m.T - np.diag((m*(m+1)).ravel() / 2) - M1
            # number of links from node n
            r = z[nn, :].T@X[nn, n][:, np.newaxis]
            R = np.tile(r, (1, K))

            beta_old_comps = betaln(M1+R+a, M0+M-R+b)-betaln(M1+a, M0+b)
            # new component's values
            beta_new_comp = betaln(r+a, m-r+b)-betaln(a, b)
            likelihood_change = sum(np.concatenate((beta_old_comps,
                                    beta_new_comp), axis=1), 0)
            # used to avoid log(0)
            m[m == 0] = eps
            prior_change = np.log(np.append(m, theta))
            # Log prob of n belonging to existing or new component
            logP = likelihood_change + prior_change
            # Convert from log probability
            P = np.exp(logP-logP.max())
            # Random component according to P
            rand_arr = np.random.rand() < np.cumsum(P)/P.sum()
            i = rand_arr.tolist().index(True)
            z[n, :] = 0
            new_arr = np.zeros((z.shape[0], 1))
            z = np.concatenate((z, new_arr), axis=1)
            z[n, i] = 1
            # remove empty components
            empty_cluster = np.argwhere(np.sum(z, axis=0) == 0).squeeze()
            z = np.delete(z, empty_cluster, 1)
        Z.append(z.copy())
    return Z
