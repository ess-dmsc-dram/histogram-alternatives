from abc import ABC, abstractmethod
from typing import Callable, Dict, Iterable, List

import numpy as np
from numpy.typing import ArrayLike
from rich.progress import BarColumn, Progress, SpinnerColumn, \
    TimeElapsedColumn, TimeRemainingColumn
import scipp as sc
from statsmodels.nonparametric.kde import KDEUnivariate

from .stats import make_bin_edges


def _data_array_1d(values, coord, dim):
    return sc.DataArray(sc.array(dims=[dim], values=values),
                        coords={dim: sc.array(dims=[dim], values=coord)})


class Estimator(ABC):
    def __init__(self, name: str, full_sample: ArrayLike, x: ArrayLike,
                 param_fit: Dict[str, Callable[[sc.DataArray], sc.Variable]]):
        self.name = name
        self.base_sample = full_sample
        self.x = x
        self.param_fit = param_fit

        self._progress = Progress(SpinnerColumn('flip'),
                                  '[progress.description]{task.description}',
                                  BarColumn(bar_width=None), 'ETA:',
                                  TimeRemainingColumn(), 'Elapsed:',
                                  TimeElapsedColumn())

    @abstractmethod
    def estimate_once(self, sample: ArrayLike, x: ArrayLike) -> sc.DataArray:
        pass

    def _do_estimate(self, sample: ArrayLike, x: ArrayLike) -> sc.DataArray:
        result = self.estimate_once(sample, x)
        for name, func in self.param_fit.items():
            result.attrs[name] = func(result)
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
        return {
            name: None
            for name in ('__average', '__square_average', *self.param_fit)
        }

    def _add_result(self, samples: Dict[str, List], result: sc.DataArray) -> None:
        samples.setdefault('__values', []).append(result.data.values)
        for name in self.param_fit:
            samples.setdefault(name, []).append(result.meta[name].value)

    def __call__(self, n_resample: int, rng: np.random.Generator) -> sc.DataArray:
        samples = {}
        coord = None
        for result in map(lambda sample: self._do_estimate(sample, self.x),
                          self._iter_samples(n_resample, rng)):
            self._add_result(samples, result)
            if coord is None:
                coord = result.coords['x']
            else:
                assert sc.allclose(coord, result.coords['x'])

        return sc.DataArray(sc.array(dims=['x'],
                                     values=np.mean(samples['__values'], axis=0),
                                     variances=np.var(samples['__values'], axis=0)),
                            coords={'x': coord},
                            attrs={
                                'sample_size': sc.scalar(len(self.base_sample)),
                                'n_resample': sc.scalar(n_resample),
                                **{
                                    name: sc.scalar(value=np.mean(samples[name]),
                                                    variance=np.var(samples[name]))
                                    for name in self.param_fit
                                }
                            })


class HistogramEstimator(Estimator):
    def __init__(self, full_sample: ArrayLike, x: ArrayLike,
                 param_fit: Dict[str, Callable[[sc.DataArray], sc.Variable]]):
        super().__init__('histogram', full_sample, x, param_fit)
        self.bin_edges = make_bin_edges(full_sample, xmin=x[0], xmax=x[-1])

    def estimate_once(self, sample: ArrayLike, _x: ArrayLike) -> sc.DataArray:
        bin_centres = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
        density = np.histogram(sample, bins=self.bin_edges, density=True)[0]
        return _data_array_1d(density, bin_centres, 'x')


class KDEEstimator(Estimator):
    def __init__(self, full_sample: ArrayLike, x: ArrayLike,
                 param_fit: Dict[str, Callable[[sc.DataArray], sc.Variable]]):
        super().__init__('KDE', full_sample, x, param_fit)

    def estimate_once(self, sample: ArrayLike, x: ArrayLike) -> sc.DataArray:
        kde = KDEUnivariate(sample)
        kde.fit()
        density = kde.evaluate(x)
        return _data_array_1d(density, x, 'x')


class MaximumLikelihoodEstimator(Estimator):
    def __init__(self, full_sample: ArrayLike, x: ArrayLike, rvs):
        if not isinstance(rvs.shapes, str):
            raise ValueError('Only RVS with a single parameter are supported')
        super().__init__('MLE',
                         full_sample,
                         x,
                         param_fit={rvs.shapes: lambda data: data.meta[rvs.shapes]})
        self.rvs = rvs

    def estimate_once(self, sample: ArrayLike, x: ArrayLike) -> sc.DataArray:
        param = self.rvs.fit(sample, floc=0.0, fscale=1.0, method='MLE')[0]
        data = _data_array_1d(self.rvs.pdf(x, param), x, 'x')
        data.attrs[self.rvs.shapes] = sc.scalar(param)
        return data
