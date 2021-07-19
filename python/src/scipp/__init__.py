# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @file
# @author Simon Heybrock

# flake8: noqa
import os

if os.name == "nt" and "CONDA_PREFIX" in os.environ:
    # Due to https://github.com/conda-forge/python-feedstock/issues/444 , combinations of Python3, Anaconda and Windows
    # don't respect os.add_dll_path(...), which is python's mechanism for setting DLL search directories. Instead we
    # need to explicitly add it to the PATH environment variable. For non-conda versions of python we want to keep using
    # the usual python mechanism.
    #
    # This is probably due to the following patch in conda-built versions of python:
    # https://github.com/conda-forge/python-feedstock/blob/289b2a8017ddd000896e525f18867f4caacec6f2/recipe/patches/0020-Add-CondaEcosystemModifyDllSearchPath.patch
    #
    import importlib.resources
    with importlib.resources.path("scipp", "__init__.py") as path:
        dll_directory = (path.parent.parent / "bin").resolve()
        os.environ["PATH"] += os.pathsep + str(dll_directory)

from . import runtime_config

user_configuration_filename = runtime_config.config_filename
config = runtime_config.load()
del runtime_config

from ._scipp import _debug_
if _debug_:
    import warnings

    def custom_formatwarning(msg, *args, **kwargs):
        return str(msg) + '\n'

    warnings.formatwarning = custom_formatwarning
    warnings.warn(
        'You are running a "Debug" build of scipp. For optimal performance use a "Release" build.'
    )

from ._scipp import __version__
# Import classes
from ._scipp.core import Variable, DataArray, Dataset, GroupByDataArray, \
                         GroupByDataset, Unit
# Import errors
from ._scipp.core import BinEdgeError, BinnedDataError, CoordError, \
                         DataArrayError, DatasetError, DimensionError, \
                         DTypeError, NotFoundError, SizeError, SliceError, \
                         UnitError, VariableError, VariancesError
# Import submodules
from ._scipp.core import units, dtype, buckets
from . import geometry
# Import functions
from ._scipp.core import choose, logical_and, logical_or, logical_xor, where
# Import python functions
from .show import show, make_svg
from .table import table
from .plotting import plot
from .extend_units import *
from .html import to_html, make_html
from .object_list import _repr_html_
from .utils import collapse, slices
from .compat.dict import to_dict, from_dict
from .sizes import _make_sizes

# Wrappers for free functions from _scipp.core
from ._arithmetic import *
from ._bins import *
from ._counts import *
from ._comparison import *
from ._cumulative import *
from ._dataset import *
from ._groupby import *
from ._math import *
from ._operations import *
from ._unary import *
from ._reduction import *
from ._shape import *
from ._trigonometry import *
from ._variable import *

setattr(Variable, '_repr_html_', make_html)
setattr(DataArray, '_repr_html_', make_html)
setattr(Dataset, '_repr_html_', make_html)

from .io.hdf5 import to_hdf5 as _to_hdf5

setattr(Variable, 'to_hdf5', _to_hdf5)
setattr(DataArray, 'to_hdf5', _to_hdf5)
setattr(Dataset, 'to_hdf5', _to_hdf5)

setattr(Variable, 'sizes', property(_make_sizes))
setattr(DataArray, 'sizes', property(_make_sizes))
setattr(Dataset, 'sizes', property(_make_sizes))

from ._bins import _bins, _set_bins, _events

setattr(Variable, 'bins', property(_bins, _set_bins))
setattr(DataArray, 'bins', property(_bins, _set_bins))
setattr(Dataset, 'bins', property(_bins, _set_bins))
setattr(Variable, 'events', property(_events))
setattr(DataArray, 'events', property(_events))

from ._structured import _fields

setattr(
    Variable, 'fields',
    property(
        _fields,
        doc=
        """Provides access to fields of structured types such as vectors or matrices."""
    ))

from ._bins import _groupby_bins

setattr(GroupByDataArray, 'bins', property(_groupby_bins))
setattr(GroupByDataset, 'bins', property(_groupby_bins))

setattr(Variable, 'plot', plot)
setattr(DataArray, 'plot', plot)
setattr(Dataset, 'plot', plot)

# Prevent unwanted conversion to numpy arrays by operations. Properly defining
# __array_ufunc__ should be possible by converting non-scipp arguments to
# variables. The most difficult part is probably mapping the ufunc to scipp
# functions.
for _cls in (Variable, DataArray, Dataset):
    setattr(_cls, '__array_ufunc__', None)
del _cls

from . import _binding

_binding.bind_get()
for _cls in (Variable, DataArray):
    _binding.bind_functions_as_methods(
        _cls, globals(), ('broadcast', 'flatten', 'fold', 'transpose', 'all',
                          'any', 'mean', 'sum', 'nanmean', 'nansum'))
del _cls
_binding.bind_functions_as_methods(
    Variable, globals(), ('cumsum', 'max', 'min', 'nanmax', 'nanmin'))
_binding.bind_functions_as_methods(DataArray, globals(), ('groupby', ))
_binding.bind_functions_as_methods(Dataset, globals(), ('groupby', ))
del _binding

from .serialization import encode, decode

try:
    import msgpack
    from typing import List, Dict, Tuple, Union
    from distributed.protocol import dask_serialize, dask_deserialize

    @dask_serialize.register(Variable)
    @dask_serialize.register(DataArray)
    def serialize(var: Union[Variable, DataArray]) -> Tuple[Dict, List[bytes]]:
        header = {}
        frames = [msgpack.packb(var, default=encode)]
        return header, frames

    @dask_deserialize.register(Variable)
    @dask_deserialize.register(DataArray)
    def deserialize(header: Dict,
                    frames: List[bytes]) -> Union[Variable, DataArray]:
        return msgpack.unpackb(frames[0], object_hook=decode)
except ImportError:
    pass
