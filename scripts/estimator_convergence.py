from typing import Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipp as sc
from halt.models import sphere, sphere_pdf
from halt.estimators import HistogramEstimator, KDEEstimator, MaximumLikelihoodEstimator

TRUE_R = 2.0
SAMPLE_SIZES = [100, 200, 300, 400]
N_RESAMPLE = 100
N_RESAMPLE_MLE = 20
COLOURS = {'KDE': 'C0', 'histogram': 'C1', 'MLE': 'C2', 'true': 'k'}


def make_sample(dist, size, rng):
    return dist.rvs(size=size, random_state=rng)


def data_array_1d(values, coord, dim):
    return sc.DataArray(sc.array(dims=[dim], values=values),
                        coords={dim: sc.array(dims=[dim], values=coord)})


def maximum_likelihood(sample):
    return sphere.fit(sample, floc=0.0, fscale=1.0, method='MLE')[0]


def unary_fit(*args, **kwargs):
    (p,), ((_e,),) = scipy.optimize.curve_fit(*args, **kwargs)
    return p


def fit_r(data: sc.DataArray) -> sc.Variable:
    return sc.scalar(
        unary_fit(sphere_pdf,
                  xdata=data.coords['x'].values,
                  ydata=data.data.values,
                  p0=[1]))


def find_xlim(densities: Iterable[sc.DataArray]) -> Tuple[float, float]:
    lo = min(map(lambda da: da.coords['x'].min().value, densities))
    hi = max(map(lambda da: da.coords['x'].max().value, densities))
    return float(lo), float(hi)


def errorstep(ax, x, y, e, c, **kwargs):
    ax.fill_between(x, y - e, y + e, step='mid', facecolor=c, alpha=0.5)
    ax.step(x, y, where='mid', c=c, **kwargs)


def errorfill(ax, x, y, e, c, **kwargs):
    ax.fill_between(x, y - e, y + e, facecolor=c, alpha=0.5)
    ax.plot(x, y, c=c, **kwargs)


def plot_densities(densities: Dict[int, Dict[str, sc.DataArray]],
                   true_distribution) -> None:
    n_plots = len(densities)
    fig, axs = plt.subplots(nrows=n_plots // 3 + min(n_plots % 3, 1), ncols=min(n_plots, 3),
                            squeeze=False, sharex='all', sharey='all',
                            gridspec_kw={'wspace': 0, 'hspace': 0})

    for ax, (n, estimated) in zip(axs.flat, densities.items()):
        x_true = np.linspace(*find_xlim(densities[n].values()), 1000)
        ax.plot(x_true, true_distribution.pdf(x_true), c=COLOURS['true'], label='pdf')
        for name, data in estimated.items():
            fn = errorstep if name == 'histogram' else errorfill
            fn(ax, data.coords['x'].values, data.data.values, np.sqrt(data.data.variances),
               c=COLOURS[name], label=name)

        ax.set_yscale('log')
        ax.set_ylim((1e-6, 2))

    for ax in axs[:, 0]:
        ax.set_ylabel('density')

    for ax in axs[-1, :]:
        ax.set_xlabel('q')

    axs[0, 0].legend()
    fig.tight_layout()


def plot_parameters(densities: Dict[int, Dict[str, sc.DataArray]]) -> None:
    fig, ax = plt.subplots(1, 1)
    ax.axhline(TRUE_R, c=COLOURS['true'])
    ax.set_xlabel('n')
    ax.set_ylabel('r')

    fit_results = {}
    for n, per_n in densities.items():
        for name, result in per_n.items():
            fit_results.setdefault(name, []).append((n, result.attrs['r'].value, result.attrs['r'].variance))
    for name, data in fit_results.items():
        x, y, e = zip(*data)
        ax.errorbar(x, y, e, c=COLOURS[name], label=name)

    ax.legend()
    fig.tight_layout()


def estimate(sample, x, rng):
    estimators = (KDEEstimator(sample, x, {'r': fit_r}),
                  HistogramEstimator(sample, x, {'r': fit_r}))
    results = {estimator.name: estimator(N_RESAMPLE, rng) for estimator in estimators}
    mle = MaximumLikelihoodEstimator(sample, x, sphere)
    results[mle.name] = mle(N_RESAMPLE_MLE, rng)
    return results


def main():
    rng = np.random.default_rng(8471)
    true_distribution = sphere(r=TRUE_R, loc=0.0, scale=1.0)
    base_sample = make_sample(true_distribution, SAMPLE_SIZES[-1], rng)
    x = np.linspace(0, 5, 1000)
    densities = {size: estimate(base_sample[:size], x, rng) for size in SAMPLE_SIZES}
    plot_densities(densities, true_distribution)
    plot_parameters(densities)
    plt.show()


if __name__ == '__main__':
    main()
