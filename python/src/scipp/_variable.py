# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Matthew Andrew
from . import units, Unit, dtype, Variable
from ._scipp import core as _cpp
from typing import Any, Sequence, Union
import numpy


def scalar(value: Any,
           variance: Any = None,
           unit: Union[Unit, str] = units.dimensionless,
           dtype: type(dtype.float64) = None) -> Variable:
    """Constructs a zero dimensional :class:`Variable` with a unit and optional
    variance.

    :seealso: :py:func:`scipp.zeros` :py:func:`scipp.ones`
              :py:func:`scipp.empty` :py:func:`scipp.array`

    :param value: Initial value.
    :param variance: Optional, initial variance, Default=None
    :param unit: Optional, unit. Default=dimensionless
    :param dtype: Optional, type of underlying data. Default=None,
      in which case type is inferred from value input.
      Cannot be specified for value types of
      str, Dataset or DataArray.
    :raises: Add this
    """
    if dtype is None:
        return Variable(value=value, variance=variance, unit=unit)
    else:
        try:
            return Variable(value=value,
                            variance=variance,
                            unit=unit,
                            dtype=dtype)
        except TypeError:
            # Raise a more comprehensible error message in the case
            # where a dtype cannot be specified.
            raise TypeError(f"Cannot convert {value} to {dtype}. "
                            f"Try omitting the 'dtype=' parameter.")


def zeros(*,
          dims: Sequence[str],
          shape: Sequence[int],
          unit: Union[Unit, str] = units.dimensionless,
          dtype: type(dtype.float64) = dtype.float64,
          variances: bool = False) -> Variable:
    """Constructs a :class:`Variable` with default initialised values with
    given dimension labels and shape.
    Optionally can add default initialised variances.
    Only keyword arguments accepted.

    :seealso: :py:func:`scipp.ones` :py:func:`scipp.empty`
              :py:func:`scipp.scalar` :py:func:`scipp.array`

    :param dims: Dimension labels.
    :param shape: Dimension sizes.
    :param unit: Optional, unit of contents. Default=dimensionless
    :param dtype: Optional, type of underlying data. Default=float64
    :param variances: Optional, boolean flag, if True includes variances
      initialised to the default value for dtype.
      For example for a float type values and variances would all be
      initialised to 0.0. Default=False
    """
    return Variable(dims=dims,
                    shape=shape,
                    unit=unit,
                    dtype=dtype,
                    variances=variances)


def zeros_like(var: Variable) -> Variable:
    """Constructs a :class:`Variable` with the same dims, shape, unit and dtype
    as the input variable, but with all values initialized to 0. If the input
    has variances, all variances in the output are set to 0.

    :seealso: :py:func:`scipp.zeros` :py:func:`scipp.ones_like`

    :param var: Input variable.
    """
    return zeros(dims=var.dims,
                 shape=var.shape,
                 unit=var.unit,
                 dtype=var.dtype,
                 variances=var.variances is not None)


def ones(*,
         dims: Sequence[str],
         shape: Sequence[int],
         unit: Union[Unit, str] = units.dimensionless,
         dtype: type(dtype.float64) = dtype.float64,
         variances: bool = False) -> Variable:
    """Constructs a :class:`Variable` with values initialized to 1 with
    given dimension labels and shape.

    :seealso: :py:func:`scipp.zeros` :py:func:`scipp.empty`
              :py:func:`scipp.scalar` :py:func:`scipp.array`

    :param dims: Dimension labels.
    :param shape: Dimension sizes.
    :param unit: Optional, unit of contents. Default=dimensionless
    :param dtype: Optional, type of underlying data. Default=float64
    :param variances: Optional, boolean flag, if True includes variances
                      initialised to 1. Default=False
    """
    return _cpp.ones(dims, shape, unit, dtype, variances)


def ones_like(var: Variable) -> Variable:
    """Constructs a :class:`Variable` with the same dims, shape, unit and dtype
    as the input variable, but with all values initialized to 1. If the input
    has variances, all variances in the output are set to 1.

    :seealso: :py:func:`scipp.ones` :py:func:`scipp.zeros_like`

    :param var: Input variable.
    """
    return ones(dims=var.dims,
                shape=var.shape,
                unit=var.unit,
                dtype=var.dtype,
                variances=var.variances is not None)


def empty(*,
          dims: Sequence[str],
          shape: Sequence[int],
          unit: Union[Unit, str] = units.dimensionless,
          dtype: type(dtype.float64) = dtype.float64,
          variances: bool = False) -> Variable:
    """Constructs a :class:`Variable` with uninitialized values with given
    dimension labels and shape. USE WITH CARE! Uninitialized means that values
    have undetermined values. Consider using :py:func:`scipp.zeros` unless you
    know what you are doing and require maximum performance.

    :seealso: :py:func:`scipp.zeros` :py:func:`scipp.ones`
              :py:func:`scipp.scalar` :py:func:`scipp.array`

    :param dims: Dimension labels.
    :param shape: Dimension sizes.
    :param unit: Optional, unit of contents. Default=dimensionless
    :param dtype: Optional, type of underlying data. Default=float64
    :param variances: Optional, boolean flag, if True includes uninitialized
                      variances. Default=False
    """
    return _cpp.empty(dims, shape, unit, dtype, variances)


def array(*,
          dims: Sequence[str],
          values: Union[numpy.ndarray, list],
          variances: Union[numpy.ndarray, list] = None,
          unit: Union[Unit, str] = units.dimensionless,
          dtype: type(dtype.float64) = None) -> Variable:
    """Constructs a :class:`Variable` with given dimensions, containing given
    values and optional variances. Dimension and value shape must match.
    Only keyword arguments accepted.

    :seealso: :py:func:`scipp.zeros` :py:func:`scipp.ones`
              :py:func:`scipp.empty` :py:func:`scipp.scalar`

    :param dims: Dimension labels.
    :param values: Initial values.
    :param variances: Optional, initial variances, must be same shape
      and size as values. Default=None
    :param unit: Optional, data unit. Default=dimensionless
    :param dtype: Optional, type of underlying data. Default=None,
      in which case type is inferred from value input.
    """
    return Variable(dims=dims,
                    values=values,
                    variances=variances,
                    unit=unit,
                    dtype=dtype)


# Wrapper to make datetime usable without importing numpy manually.
datetime64 = numpy.datetime64
