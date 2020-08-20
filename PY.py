from scipy.special import digamma, gammaln, gamma, betaln
import random
from math import exp
import numpy as np


def f(theta, alpha, n, p):
    return exp(gammaln(theta + n + alpha) - gammaln(theta + n))*exp(gammaln(theta + 1) - gammaln(theta + alpha)) * (1 / alpha) - (theta/alpha) - p


def df(theta, alpha, n):
    return exp(gammaln(theta + n + alpha) - gammaln(theta + n)) * exp(gammaln(theta + 1) - gammaln(theta + alpha)) * ((digamma(theta + n + alpha) + digamma(theta + 1) - digamma(theta + n) - digamma(theta + alpha)) / alpha) - (1 / alpha)


def dx(f, theta, alpha, n, p):
    return abs(0-f(theta, alpha, n, p))


def newtons_method(f, df, theta0, n, p, e, max_iter):
    alpha = 1
    i = 0
    max_iter_fixed_alpha = max_iter // 3
    # Theta cannot be lower than - alpha
    while (alpha == 1 or theta0 <= - alpha) and i < max_iter:
        # Get random alpha
        alpha = random.uniform(0, 1)
        delta = dx(f, theta0, alpha, n, p)
        j = 0
        while delta > e and j < max_iter_fixed_alpha:
            theta0 = theta0 - f(theta0, alpha, n, p)/df(theta0, alpha, n)
            delta = dx(f, theta0, alpha, n, p)
            i += 1
            j += 1
    if i >= max_iter:
        raise ValueError("reached max_iter, consider increasing")
    return theta0, alpha, f(theta0, alpha, n, p)


def crp_parameters(n_politicians, n_parties, max_iter):
    e = 1e-9
    max_iter = 1000000
    theta0 = 1
    n = n_politicians
    p = n_parties
    theta, alpha, value = newtons_method(f, df, theta0, n, p, e, max_iter)
    print(f"CRP parameters: theta: {theta}, alpha: {alpha}")
    print(f"value of the function: {value}")
    return theta, alpha


def beta_bernoulli_irm(X, a=1, b=1, theta=6, alpha=0.5,  T=100):

    # number of nodes in the network
    N = X.shape[0]
    z = np.ones((N, 1))
    Z = []
    np.fill_diagonal(X, 0)
    eps = 1e-200

    for _ in range(T):
        # component of each node (at the beginning all nodes belong to the same component)
        for n in range(N):
            # indices considered
            nn = [x for x in range(N) if x != n]
            # number of components
            K = z.shape[1]
            # number of nodes for each components
            m = sum(z[nn, :])[:, np.newaxis]
            M = np.tile(m, (1, K))

            # X_nn_nn in matlab is X[nn, nn]
            X_nn_nn = np.delete(X, n, 0)
            X_nn_nn = np.delete(X_nn_nn, n, 1)
            # number of links between components
            M1 = z[nn, :].T@X_nn_nn@z[nn, :] - np.diag(sum(X_nn_nn@z[nn, :]*z[nn, :])/2)
            # number of no-links between components
            M0 = m@m.T - np.diag((m*(m+1)).ravel() / 2) - M1
            # number of links from node n(0)
            r = z[nn, :].T@X[nn, n][:, np.newaxis]
            R = np.tile(r, (1, K))
            beta_old_comps = betaln(M1+R+a, M0+M-R+b)-betaln(M1+a, M0+b)
            # new component's values
            beta_new_comp = betaln(r+a, m-r+b)-betaln(a, b)
            likelihood_change = sum(np.concatenate((beta_old_comps,
                                    beta_new_comp), axis=1), 0)


            # PY and DP difference
            prior_change_existing_partition = m - alpha
            # used to avoid log(<0)
            prior_change_existing_partition[prior_change_existing_partition < 0] = eps
            prior_change_new_partition = theta + (alpha*K)
            prior_change = np.log(np.append(prior_change_existing_partition, prior_change_new_partition))
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
        print(f"sampling: {_}")
    return Z
