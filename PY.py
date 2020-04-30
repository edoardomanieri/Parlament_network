from scipy.special import digamma, polygamma, gamma, gammaln
import random
from math import exp


def truef(theta, alpha, n, p):
    return ((gamma(theta + n + alpha)*gamma(theta + 1)) / (alpha * gamma(theta + n) * gamma(theta + alpha))) - (theta/alpha) - p


def f(theta, alpha, n, p):
    return exp(gammaln(theta + n + alpha) - gammaln(theta + n))*exp(gammaln(theta + 1) - gammaln(theta + alpha))* (1 / alpha) - (theta/alpha) - p


def truedf(theta, alpha, n):
    return (gamma(theta + n + alpha) * gamma(theta + 1) * (digamma(theta + n + alpha) + digamma(theta + 1) - digamma(theta + n) - digamma(theta + alpha))) / (alpha * gamma(theta + n) * gamma(theta + alpha)) - 1 / alpha


def df(theta, alpha, n):
    return exp(gammaln(theta + n + alpha) - gammaln(theta + n)) * exp(gammaln(theta + 1) - gammaln(theta + alpha)) * (digamma(theta + n + alpha) + digamma(theta + 1) - digamma(theta + n) - digamma(theta + alpha)) / alpha - (1 / alpha)


def dx(f, theta, alpha, n, p):
    return abs(0-f(theta, alpha, n, p))


def newtons_method(f, df, theta0, n, p, e, max_iter):
    alpha = 1
    while alpha == 1 or theta0 <= - alpha:
        alpha = random.uniform(0, 1)
        delta = dx(f, theta0, alpha, n, p)
        i = 0
        while delta > e and i < max_iter:
            theta0 = theta0 - f(theta0, alpha, n, p)/df(theta0, alpha, n)
            delta = dx(f, theta0, alpha, n, p)
            i += 1
        if i >= max_iter:
            print("reached max_iter, consider increasing")
    return theta0, alpha, f(theta0, alpha, n, p)


def crp_parameters(n_politicians, n_parties, max_iter):
    e = 1e-9
    max_iter = 200000
    theta0 = 1
    n = n_politicians
    p = n_parties
    theta, alpha, value = newtons_method(f, df, theta0, n, p, e, max_iter)
    print(f"CRP parameters: theta: {theta}, alpha: {alpha}")
    print(f"value of the function: {value}")
    return theta, alpha
