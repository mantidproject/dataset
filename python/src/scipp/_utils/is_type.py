# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @file
# @author Neil Vaytet

from .._scipp import core as sc


def is_variable(obj):
    """
    Return True if the input is of type scipp.core.Variable.
    """
    return isinstance(obj, sc.Variable) or isinstance(obj, sc.VariableView)


def is_scalar(obj):
    """
    Return True if the input is a scalar
    """
    return obj.dims == []


def is_dataset(obj):
    """
    Return True if the input is of type scipp.core.Variable.
    """
    return isinstance(obj, sc.Dataset) or isinstance(obj, sc.DatasetView)


def is_data_array(obj):
    """
    Return True if the input is of type scipp.core.Variable.
    """
    return isinstance(obj, sc.DataArray) or isinstance(obj, sc.DataArrayView)


def is_dataset_or_array(obj):
    """
    Return True if the input object is either a Dataset or DataArray.
    """
    return is_dataset(obj) or is_data_array(obj)
