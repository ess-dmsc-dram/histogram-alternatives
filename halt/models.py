import math

from numba import vectorize
import numpy as np
import scipy
import scipy.stats


@vectorize(['float64(float64, float64)'])
def _sphere_pdf(q, r):
    N = 3 * np.pi / 5 / q
    if r == 0:
        return 1 / N
    qr = q * r
    return 9 / N * (math.sin(qr) - qr * math.cos(qr))**2 / qr**6


@vectorize(['float64(float64, float64)'])
def _sphere_cdf_impl(qr, si):
    return (4 * qr**5 * si - 5 * qr**2 + (qr**2 + 6) * qr * np.sin(2 * qr) +
            (2 * qr**4 - qr**2 + 3) * np.cos(2 * qr) - 3) / (2 * np.pi * qr**5)


def _sphere_cdf(q, r):
    qr = q * r
    si = scipy.special.sici(2 * qr)[0]
    return _sphere_cdf_impl(qr, si)


class Sphere(scipy.stats.rv_continuous):
    """
    Random variable the scattering intensity of a sphere.

    f(x) ~ [3 (sin(qr) - qr cos(qr)) / (qr)^3]^2

    See for details https://www.sasview.org/docs/user/models/sphere.html
    """
    def __init__(self, r):
        super().__init__(a=0)
        self.r = r

    def _pdf(self, x):
        return _sphere_pdf(x, self.r)

    def _cdf(self, x):
        return _sphere_cdf(x, self.r)
