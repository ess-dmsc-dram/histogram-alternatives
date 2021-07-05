import numpy as np


def expectation_value(x, p, normed=False):
    e = np.sum(x * p)
    if normed:
        return e
    return e / np.sum(p)


def moments(x, p, mom, normed=False):
    if not normed:
        norm = np.sum(p)
        if np.abs(norm) > 1e-13:
            p = p / norm

    res = dict()
    m = expectation_value(x, p, normed=True)
    if 'mean' in mom:
        res['mean'] = m
        if len(mom) == 1:
            return res
        
    x_minus_m = x - m
    x_minus_m_sq = x_minus_m ** 2
    var = expectation_value(x_minus_m_sq, p, normed=True)
    if 'variance' in mom:
        res['variance'] = var
        
    if 'skewness' in mom:
        res['skewness'] = expectation_value(x_minus_m_sq * x_minus_m / var / np.sqrt(var),
                                            p, normed=True)
        
    if 'kurtosis' in mom:
        res['kurtosis'] = expectation_value((x_minus_m_sq / var)**2,
                                            p, normed=True)
        
    return res
    
    
moments.all = ('mean', 'variance', 'skewness', 'kurtosis')


def mean(x, p, normed=False):
    return moments(x, p, mom=('mean',), normed=normed)['mean']


def variance(x, p, m=None, normed=False):
    return moments(x, p, mom('variance',), normed=normed)['variance']
