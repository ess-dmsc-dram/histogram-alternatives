from pathlib import Path
from tempfile import gettempdir

import scipp as sc
import scippneutron as scn


def _cache_filename(base_filename, dataset):
    dataset_part = ('__' + dataset) if dataset else ''
    return Path(gettempdir()) / ('cache__' + base_filename.stem + dataset_part + '.h5')


def cached_load(filename, dataset=None, replace_cache=False):
    filename = Path(filename)
    cache_filename = _cache_filename(filename, dataset)
    if cache_filename.exists() and not replace_cache:
        return sc.io.open_hdf5(cache_filename)
    data = scn.load(str(filename))
    if dataset:
        data = data[dataset]
    data.to_hdf5(cache_filename)
    return data
