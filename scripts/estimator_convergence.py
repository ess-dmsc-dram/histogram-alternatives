from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
import scipy
from rich.progress import BarColumn, Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
import scipp as sc
from statsmodels.nonparametric.kde import KDEUnivariate

from halt.models import sphere, sphere_pdf
from halt.stats import make_bin_edges

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


def fit_model(q, r):
    return sphere_pdf(q, r)


def unary_fit(*args, **kwargs):
    (p,), ((_e,),) = scipy.optimize.curve_fit(*args, **kwargs)
    return p


class SphereFitMixin:
    @staticmethod
    def fit_parameters(data: sc.DataArray) -> Dict[str, sc.Variable]:
        if 'r' in data.meta:
            return {'r': data.meta['r']}
        return {'r': sc.scalar(unary_fit(sphere_pdf, xdata=data.coords['x'].values,
                                         ydata=data.data.values, p0=[1]))}


class Estimator(ABC):
    def __init__(self, name: str, full_sample: ArrayLike, x: ArrayLike,
                 param_names: Sequence[str]):
        self.name = name
        self.base_sample = full_sample
        self.x = x
        self.param_names = tuple(param_names)

        self._progress = Progress(SpinnerColumn('flip'),
                                  '[progress.description]{task.description}',
                                  BarColumn(bar_width=None),
                                  'ETA:',
                                  TimeRemainingColumn(),
                                  'Elapsed:',
                                  TimeElapsedColumn())

    @abstractmethod
    def estimate_once(self, sample: ArrayLike, x: ArrayLike) -> sc.DataArray:
        pass

    def _do_estimate(self, sample: ArrayLike, x: ArrayLike) -> sc.DataArray:
        result = self.estimate_once(sample, x)
        if hasattr(self, 'fit_parameters'):
            for key, val in self.fit_parameters(result).items():
                result.attrs[key] = val
        return result

    def resample(self, rng: np.random.Generator) -> ArrayLike:
        return rng.choice(self.base_sample, size=len(self.base_sample), replace=True)

    def _tracked_range(self, n: int) -> Iterable[int]:
        with self._progress:
            yield from self._progress.track(
                range(n),
                description=f'Estimating {self.name} (size={len(self.base_sample)})')

    def _iter_samples(self, n: int, rng: np.random.Generator) -> Iterable[ArrayLike]:
        for _ in self._tracked_range(n):
            yield self.resample(rng)

    def _make_accumulators(self) -> Dict:
        return {name: None for name in ('__average', '__square_average',
                                        *self.param_names)}

    def _add_result(self, accumulators: Dict[str, sc.DataArray], result: sc.DataArray) -> None:
        def add_value(key, value):
            if key in accumulators:
                accumulators[key] += value
            else:
                accumulators[key] = deepcopy(value)

        add_value('__average', result.data)
        add_value('__square_average', result.data ** 2)
        for name in self.param_names:
            add_value(name, result.meta[name])

    def __call__(self, n_resample: int, rng: np.random.Generator) -> sc.DataArray:
        accumulators = dict()
        coord = None
        for result in map(lambda sample: self._do_estimate(sample, self.x),
                          self._iter_samples(n_resample, rng)):
            self._add_result(accumulators, result)
            if coord is None:
                coord = result.coords['x']
            else:
                assert sc.allclose(coord, result.coords['x'])

        data = accumulators['__average']
        data.variances = (data.values ** 2 - accumulators['__square_average'].values) / (n_resample - 1)
        data.values /= n_resample
        return sc.DataArray(data, coords={'x': coord},
                            attrs={'sample_size': sc.scalar(len(self.base_sample)),
                                   'n_resample': sc.scalar(n_resample)})


class HistogramEstimator(Estimator, SphereFitMixin):
    def __init__(self, full_sample: ArrayLike, x: ArrayLike):
        super().__init__('histogram', full_sample, x, param_names=('r',))
        self.bin_edges = make_bin_edges(full_sample, xmin=x[0], xmax=x[-1])

    def estimate_once(self, sample: ArrayLike, _x: ArrayLike) -> sc.DataArray:
        bin_centres = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
        density = np.histogram(sample, bins=self.bin_edges, density=True)[0]
        return data_array_1d(density, bin_centres, 'x')


class KDEEstimator(Estimator, SphereFitMixin):
    def __init__(self, full_sample: ArrayLike, x: ArrayLike):
        super().__init__('KDE', full_sample, x, param_names=('r',))

    def estimate_once(self, sample: ArrayLike, x: ArrayLike) -> sc.DataArray:
        kde = KDEUnivariate(sample)
        kde.fit()
        density = kde.evaluate(x)
        return data_array_1d(density, x, 'x')


class MaximumLikelihoodEstimator(Estimator, SphereFitMixin):
    def __init__(self, full_sample: ArrayLike, x: ArrayLike):
        super().__init__('MLE', full_sample, x, param_names=('r',))

    def estimate_once(self, sample: ArrayLike, x: ArrayLike) -> sc.DataArray:
        r = maximum_likelihood(sample)
        data = data_array_1d(sphere.pdf(x, r=r), x, 'x')
        data.attrs['r'] = sc.scalar(r)
        return data


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


def estimate(sample, x, rng):
    estimators = (KDEEstimator(sample, x),
                  HistogramEstimator(sample, x))
    results = {estimator.name: estimator(N_RESAMPLE, rng) for estimator in estimators}
    mle = MaximumLikelihoodEstimator(sample, x)
    results[mle.name] = mle(N_RESAMPLE_MLE, rng)
    return results


def fit_r(densities: Dict[int, Dict[str, sc.DataArray]]) -> None:
    for ds in densities.values():
        for density in ds.values():
            density.attrs['r'] = sc.scalar(unary_fit(fit_model,
                                                     xdata=density.coords['x'].values,
                                                     ydata=density.data.values,
                                                     p0=[1]))


def main():
    rng = np.random.default_rng(8471)
    true_distribution = sphere(r=TRUE_R, loc=0.0, scale=1.0)
    base_sample = make_sample(true_distribution, SAMPLE_SIZES[-1], rng)
    x = np.linspace(0, 5, 1000)
    densities = {size: estimate(base_sample[:size], x, rng) for size in SAMPLE_SIZES}
    fit_r(densities)
    plot_densities(densities, true_distribution)
    plt.show()


if __name__ == '__main__':
    main()
