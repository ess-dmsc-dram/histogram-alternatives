import numpy.typing
import numpy as np
from scipy import stats


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
        res['kurtosis'] = expectation_value((x_minus_m_sq / var) ** 2,
                                            p, normed=True)

    return res


moments.all = ('mean', 'variance', 'skewness', 'kurtosis')


def mean(x, p, normed=False):
    return moments(x, p, mom=('mean',), normed=normed)['mean']


def variance(x, p, normed=False):
    return moments(x, p, mom=('variance',), normed=normed)['variance']


def make_bin_edges(sample: np.typing.ArrayLike,
                   xmin: float,
                   xmax: float) -> np.ndarray:
    """
    Build bins for histogramming using the Freedman–Diaconis rule.
    """
    width = 2 * stats.iqr(sample) / len(sample) ** (1 / 3)
    return np.arange(xmin, xmax + width, width)
