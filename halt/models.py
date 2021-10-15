import math

from numba import vectorize
import numpy as np
import scipy
import scipy.stats


@vectorize(['float64(float64, float64)'])
def sphere_pdf(q, r):
    N = 3 * np.pi / 5 / r
    if q == 0:
        return 1 / N
    qr = q * r
    return 9 / N * (math.sin(qr) - qr * math.cos(qr))**2 / qr**6


@vectorize(['float64(float64, float64)'])
def _sphere_cdf_impl(qr, si):
    return (4 * qr**5 * si - 5 * qr**2 + (qr**2 + 6) * qr * np.sin(2 * qr) +
            (2 * qr**4 - qr**2 + 3) * np.cos(2 * qr) - 3) / (2 * np.pi * qr**5)


def sphere_cdf(q, r):
    qr = q * r
    si = scipy.special.sici(2 * qr)[0]
    return _sphere_cdf_impl(qr, si)


class Sphere_gen(scipy.stats.rv_continuous):
    """
    Random variable the scattering intensity of a sphere.

    f(x) ~ [3 (sin(qr) - qr cos(qr)) / (qr)^3]^2

    See for details https://www.sasview.org/docs/user/models/sphere.html
    """
    def _pdf(self, x, r):
        return sphere_pdf(x, r)

    def _cdf(self, x, r):
        return sphere_cdf(x, r)


sphere = Sphere_gen(a=0.0, shapes='r', name='sphere')