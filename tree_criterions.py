import numpy as np

__all__ = ['entropy', 'gini', 'variance', 'mad_median']


def entropy(y):
    n = y.shape[0]
    unique = np.unique(y)
    entropy_val = 0.
    for cl in unique:
        p_i = y[y == cl].size / n
        entropy_val += -p_i * np.log2(p_i)
    return entropy_val


def gini(y):
    n = y.shape[0]
    unique = np.unique(y)
    gini_val = 1.
    for cl in unique:
        p_i = y[y == cl].size / n
        gini_val -= p_i ** 2
    return gini_val


def variance(y):
    return np.var(y)


def mad_median(y):
    return np.abs(y - np.median(y)).mean()
