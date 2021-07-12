# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Andrew
# flake8: noqa: E501

from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Tuple, Union

from ._scipp import core as _cpp
from ._cpp_wrapper_util import call_func as _call_cpp_func
from .typing import VariableLike


def broadcast(x: _cpp.Variable, dims: Union[List[str], Tuple[str]],
              shape: Sequence[int]) -> _cpp.Variable:
    """Broadcast a variable.

    Note that scipp operations broadcast automatically, so using this function
    directly is rarely required.

    :param x: Variable to broadcast.
    :param dims: List of new dimensions.
    :param shape: New extents in each dimension.
    :return: New variable with requested dimension labels and shape.
    """
    return _call_cpp_func(_cpp.broadcast, x, dims, shape)


def concatenate(x: VariableLike, y: VariableLike, dim: str) -> VariableLike:
    """Concatenate input arrays along the given dimension.

    Concatenation can happen in two ways:

    - Along an existing dimension, yielding a new dimension extent
      given by the sum of the input's extents.
    - Along a new dimension that is not contained in either of the inputs,
      yielding an output with one extra dimensions.

    In the case of a data array or dataset, the coords, and masks are also
    concatenated.
    Coords, and masks for any but the given dimension are required to match
    and are copied to the output without changes.

    :param x: Left hand side input.
    :param y: Right hand side input.
    :param dim: Dimension along which to concatenate.
    :raises: If the dtype or unit does not match, or if the
             dimensions and shapes are incompatible.
    :return: The absolute values of the input.

    Examples:

      >>> a = sc.arange('x', 3)
      >>> b = 100 * sc.arange('x', 3)
      >>> c = sc.concatenate(a, b, dim='x')
      >>> c
      <scipp.Variable> (x: 6)      int64  [dimensionless]  [0, 1, ..., 100, 200]
      >>> c.values
      array([  0,   1,   2,   0, 100, 200])

      >>> d = sc.concatenate(a, b, dim='y')
      >>> d
      <scipp.Variable> (y: 2, x: 3)      int64  [dimensionless]  [0, 1, ..., 100, 200]
      >>> d.values
      array([[  0,   1,   2],
             [  0, 100, 200]])
       
      >>> x = sc.DataArray(sc.arange('x', 3), coords={'x': sc.arange('x', 3)})
      >>> y = sc.DataArray(100 * sc.arange('x', 3), coords={'x': 100 * sc.arange('x', 3)})
      >>> z = sc.concatenate(x, y, dim='x')
      >>> z
      <scipp.DataArray>
      Dimensions: Sizes[x:6, ]
      Coordinates:
        x                           int64  [dimensionless]  (x)  [0, 1, ..., 100, 200]
      Data:
                                    int64  [dimensionless]  (x)  [0, 1, ..., 100, 200]
      >>> z.values
      array([  0,   1,   2,   0, 100, 200])
    """
    return _call_cpp_func(_cpp.concatenate, x, y, dim)


def fold(x: VariableLike,
         *,
         dim: str,
         sizes: Optional[Dict[str, int]] = None,
         dims: Optional[Union[List[str], Tuple[str]]] = None,
         shape: Optional[Sequence[int]] = None) -> VariableLike:
    """Fold a single dimension of a variable or data array into multiple dims.

    :param x: Variable or DataArray to fold.
    :param dim: A single dim label that will be folded into more dims.
    :param sizes: A dict mapping new dims to new shapes.
    :param dims: A list of new dims labels.
    :param shape: A list of new dim shapes.
    :raises: If the volume of the old shape is not equal to the
             volume of the new shape.
    :return: Variable or DataArray with requested dimension labels and shape.

    Examples:

      >>> v = sc.arange('x', 6)
      >>> v
      <scipp.Variable> (x: 6)      int64  [dimensionless]  [0, 1, ..., 4, 5]
      >>> sc.fold(v, dim='x', sizes={'y': 2, 'z': 3})
      <scipp.Variable> (y: 2, z: 3)      int64  [dimensionless]  [0, 1, ..., 4, 5]
      >>> sc.fold(v, dim='x', sizes={'y': 2, 'z': 3}).values
      array([[0, 1, 2],
             [3, 4, 5]])

      >>> sc.fold(v, dim='x', dims=['y', 'z'], shape=[2, 3])
      <scipp.Variable> (y: 2, z: 3)      int64  [dimensionless]  [0, 1, ..., 4, 5]

      >>> a = sc.DataArray(0.1 * sc.arange('x', 6), coords={'x': sc.arange('x', 6)})
      >>> sc.fold(a, dim='x', sizes={'y': 2, 'z': 3})
      <scipp.DataArray>
      Dimensions: Sizes[y:2, z:3, ]
      Coordinates:
        x                           int64  [dimensionless]  (y, z)  [0, 1, ..., 4, 5]
      Data:
                                  float64  [dimensionless]  (y, z)  [0.000000, 0.100000, ..., 0.400000, 0.500000]
      >>> sc.fold(a, dim='x', sizes={'y': 2, 'z': 3}).data.values
      array([[0. , 0.1, 0.2],
             [0.3, 0.4, 0.5]])
      >>> sc.fold(a, dim='x', sizes={'y': 2, 'z': 3}).coords['x'].values
      array([[0, 1, 2],
             [3, 4, 5]])
    """
    if sizes is not None:
        if (dims is not None) or (shape is not None):
            raise RuntimeError(
                "If sizes is defined, dims and shape must be None in fold.")
    else:
        if (dims is None) or (shape is None):
            raise RuntimeError("Both dims and shape must be defined in fold.")

    if dims is None:
        return _call_cpp_func(_cpp.fold, x, dim, sizes)
    else:
        return _call_cpp_func(_cpp.fold, x, dim, dict(zip(dims, shape)))


def flatten(
    x: VariableLike,
    *,
    dim: str,
    dims: Optional[Union[List[str], Tuple[str]]] = None,
) -> VariableLike:
    """Flatten multiple dimensions of a variable or data array into a single
    dimension. If dims is omitted, then we flatten all of the inputs dimensions
    into a single dim.

    :param x: Variable or DataArray to flatten.
    :param dims: A list of dim labels that will be flattened.
    :param dim: A single dim label for the resulting flattened dim.
    :raises: If the bin edge coordinates cannot be stitched back together.
    :return: Variable or DataArray with requested dimension labels and shape.

    Examples:

      >>> v = sc.array(dims=['x', 'y'], values=np.arange(6).reshape(2, 3))
      >>> v
      <scipp.Variable> (x: 2, y: 3)      int64  [dimensionless]  [0, 1, ..., 4, 5]
      >>> sc.flatten(v, dim='u')
      <scipp.Variable> (u: 6)      int64  [dimensionless]  [0, 1, ..., 4, 5]
      >>> sc.flatten(v, dims=['x', 'y'], dim='u')
      <scipp.Variable> (u: 6)      int64  [dimensionless]  [0, 1, ..., 4, 5]

      >>> v = sc.array(dims=['x', 'y', 'z'], values=np.arange(24).reshape(2, 3, 4))
      >>> v
      <scipp.Variable> (x: 2, y: 3, z: 4)      int64  [dimensionless]  [0, 1, ..., 22, 23]
      >>> sc.flatten(v, dim='u')
      <scipp.Variable> (u: 24)      int64  [dimensionless]  [0, 1, ..., 22, 23]
      >>> sc.flatten(v, dims=['x', 'y'], dim='u')
      <scipp.Variable> (u: 6, z: 4)      int64  [dimensionless]  [0, 1, ..., 22, 23]
      >>> sc.flatten(v, dims=['y', 'z'], dim='u')
      <scipp.Variable> (x: 2, u: 12)      int64  [dimensionless]  [0, 1, ..., 22, 23]

      >>> a = sc.DataArray(0.1 * sc.array(dims=['x', 'y'], values=np.arange(6).reshape(2, 3)),
      ...        coords={'x': sc.arange('x', 2),
      ...                'y': sc.arange('y', 3),
      ...                'xy': sc.array(dims=['x', 'y'],
      ...                               values=np.arange(6).reshape(2, 3))})
      >>> a
      <scipp.DataArray>
      Dimensions: Sizes[x:2, y:3, ]
      Coordinates:
        x                           int64  [dimensionless]  (x)  [0, 1]
        xy                          int64  [dimensionless]  (x, y)  [0, 1, ..., 4, 5]
        y                           int64  [dimensionless]  (y)  [0, 1, 2]
      Data:
                                  float64  [dimensionless]  (x, y)  [0.000000, 0.100000, ..., 0.400000, 0.500000]
      >>> sc.flatten(a, dim='u')
      <scipp.DataArray>
      Dimensions: Sizes[u:6, ]
      Coordinates:
        x                           int64  [dimensionless]  (u)  [0, 0, ..., 1, 1]
        xy                          int64  [dimensionless]  (u)  [0, 1, ..., 4, 5]
        y                           int64  [dimensionless]  (u)  [0, 1, ..., 1, 2]
      Data:
                                  float64  [dimensionless]  (u)  [0.000000, 0.100000, ..., 0.400000, 0.500000]

    """
    if dims is None:
        dims = x.dims
    return _call_cpp_func(_cpp.flatten, x, dims, dim)


def transpose(
        x: VariableLike,
        dims: Optional[Union[List[str], Tuple[str]]] = None) -> VariableLike:
    """Transpose dimensions of a variable, an data array, or a dataset.

    :param x: Object to transpose.
    :param dims: List of dimensions in desired order. If default,
                 reverses existing order.
    :raises: If the dtype or unit does not match, or if the
             dimensions and shapes are incompatible.
    :return: The absolute values of the input.
    """
    return _call_cpp_func(_cpp.transpose, x, dims if dims is not None else [])
