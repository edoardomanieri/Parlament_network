from scipy.special import digamma, polygamma, gamma


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