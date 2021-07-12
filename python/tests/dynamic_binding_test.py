# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @file
# @author Jan-Lukas Wynen

import pytest

import numpy as np
import scipp as sc


def make_containers():
    rng = np.random.default_rng(87325)
    var = sc.array(dims=['x', 'y'], values=rng.uniform(-5, 5, (4, 3)))
    da = sc.DataArray(var.copy(),
                      coords={
                          'x': sc.arange('x', 4),
                          'y': 0.1 * sc.arange('y', 3)
                      })
    return var, da


@pytest.mark.parametrize('func_name',
                         ('cumsum', 'max', 'mean', 'min', 'nanmax', 'nanmean',
                          'nanmin', 'nansum', 'sum'))
def test_bound_methods_reduction_variable(func_name):
    var, _ = make_containers()
    func = getattr(sc, func_name)
    assert sc.identical(getattr(var, func_name)(), func(var))


@pytest.mark.parametrize('func_name', ('any', 'all'))
def test_bound_methods_reduction_variable_bool(func_name):
    rng = np.random.default_rng(87415)
    var = sc.array(dims=['x', 'y'], values=rng.choice([True, False], (4, 3)))
    func = getattr(sc, func_name)
    assert sc.identical(getattr(var, func_name)(), func(var))


@pytest.mark.parametrize('func_name', ('mean', 'nanmean', 'nansum', 'sum'))
def test_bound_methods_reduction_dataarray(func_name):
    _, da = make_containers()
    func = getattr(sc, func_name)
    assert sc.identical(getattr(da, func_name)(), func(da))


def test_bound_methods_shape():
    var, da = make_containers()
    assert sc.identical(var.broadcast(['x', 'y', 'z'], [4, 3, 2]),
                        sc.broadcast(var, ['x', 'y', 'z'], [4, 3, 2]))
    assert sc.identical(var.transpose(['y', 'x']),
                        sc.transpose(var, ['y', 'x']))
    for obj in (var, da):
        assert sc.identical(obj.flatten(dims=['x', 'y'], dim='z'),
                            sc.flatten(obj, dims=['x', 'y'], dim='z'))
        assert sc.identical(obj.fold(dim='x', sizes={
            'a': 2,
            'b': 2
        }), sc.fold(obj, dim='x', sizes={
            'a': 2,
            'b': 2
        }))


def test_bound_methods_groupby():
    rng = np.random.default_rng(1491)
    _, da = make_containers()
    da.coords['x'] = sc.array(dims=['x'],
                              values=rng.choice([0, 1], da.coords['x'].shape))
    assert sc.identical(da.groupby('x').sum('x'), sc.groupby(da, 'x').sum('x'))

    ds = sc.Dataset(data={'item': da})
    assert sc.identical(ds.groupby('x').sum('x'), sc.groupby(ds, 'x').sum('x'))
