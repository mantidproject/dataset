# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @file
# @author Simon Heybrock
import numpy as np
import pytest

import scipp as sc


def test_shape():
    a = sc.scalar(1)
    d = sc.Dataset(data={'a': a})
    assert d.shape == []
    a = sc.empty(dims=['x'], shape=[2])
    b = sc.empty(dims=['y', 'z'], shape=[3, 4])
    d = sc.Dataset(data={'a': a, 'b': b})
    assert not bool(set(d.shape) - {2, 3, 4})


def test_sizes():
    d = sc.Dataset(data={'a': sc.scalar(value=1)})
    assert d.sizes == {}
    a = sc.empty(dims=['x'], shape=[2])
    b = sc.empty(dims=['y', 'z'], shape=[3, 4])
    d = sc.Dataset(data={'a': a, 'b': b})
    assert d.sizes == {'x': 2, 'y': 3, 'z': 4}


def test_create_empty():
    d = sc.Dataset()
    assert len(d) == 0
    assert len(d.coords) == 0
    assert len(d.dims) == 0


def test_create():
    x = sc.Variable(dims=['x'], values=np.arange(3))
    y = sc.Variable(dims=['y'], values=np.arange(4))
    xy = sc.Variable(dims=['x', 'y'], values=np.arange(12).reshape(3, 4))
    d = sc.Dataset(data={'xy': xy, 'x': x}, coords={'x': x, 'y': y})
    assert len(d) == 2
    assert sc.identical(d.coords['x'], x)
    assert sc.identical(d.coords['y'], y)
    assert sc.identical(d['xy'].data, xy)
    assert sc.identical(d['x'].data, x)
    assert set(d.dims) == {'y', 'x'}


def test_create_from_data_array():
    var = sc.Variable(dims=['x'], values=np.arange(4))
    da = sc.DataArray(var, coords={'x': var, 'aux': var})
    d = sc.Dataset(data={da.name: da})
    assert sc.identical(d[''], da)


def test_create_from_data_arrays():
    var1 = sc.Variable(dims=['x'], values=np.arange(4))
    var2 = sc.Variable(dims=['x'], values=np.ones(4))
    da1 = sc.DataArray(var1, coords={'x': var1, 'aux': var2})
    da2 = sc.DataArray(var1, coords={'x': var1, 'aux': var2})
    d = sc.Dataset()
    d['a'] = da1
    d['b'] = da2
    assert sc.identical(
        d,
        sc.Dataset(data={
            'a': var1,
            'b': var1
        },
                   coords={
                       'x': var1,
                       'aux': var2
                   }))


def test_create_from_data_array_and_variable_mix():
    var_1 = sc.Variable(dims=['x'], values=np.arange(4))
    var_2 = sc.Variable(dims=['x'], values=np.arange(4))
    da = sc.DataArray(data=var_1, coords={'x': var_1, 'aux': var_1})
    d = sc.Dataset(data={'array': da, 'variable': var_2})
    assert sc.identical(d['array'], da)
    assert sc.identical(d['variable'].data, var_2)


def test_create_with_data_array_and_additional_coords():
    var = sc.Variable(dims=['x'], values=np.arange(4))
    coord = sc.Variable(dims=['x'], values=np.arange(4))
    da = sc.DataArray(data=var, coords={'x': var, 'aux': var})
    d = sc.Dataset(data={'array': da}, coords={'y': coord})
    da.coords['y'] = coord
    assert sc.identical(d['array'], da)
    assert sc.identical(d.coords['y'], coord)
    assert sc.identical(d.coords['x'], var)
    assert sc.identical(d.coords['aux'], var)


def test_clear():
    d = sc.Dataset()
    d['a'] = sc.Variable(dims=['x'], values=np.arange(3))
    assert 'a' in d
    d.clear()
    assert len(d) == 0


def test_setitem():
    d = sc.Dataset()
    d['a'] = sc.scalar(1.0)
    assert len(d) == 1
    assert sc.identical(d['a'].data, sc.scalar(1.0))
    assert len(d.coords) == 0


def test_del_item():
    d = sc.Dataset()
    d['a'] = sc.scalar(1.0)
    assert 'a' in d
    del d['a']
    assert 'a' not in d


def test_del_item_missing():
    d = sc.Dataset()
    with pytest.raises(RuntimeError):
        del d['not an item']


def test_coord_setitem():
    var = sc.Variable(dims=['x'], values=np.arange(4))
    d = sc.Dataset(data={'a': var}, coords={'x': var})
    with pytest.raises(RuntimeError):
        d['x', 2:3].coords['y'] = sc.scalar(1.0)
    assert 'y' not in d.coords
    d.coords['y'] = sc.scalar(1.0)
    assert len(d) == 1
    assert len(d.coords) == 2
    assert sc.identical(d.coords['y'], sc.scalar(1.0))


def test_contains_coord():
    d = sc.Dataset()
    assert 'x' not in d.coords
    d.coords['x'] = sc.scalar(1.0)
    assert 'x' in d.coords


def test_coords_keys():
    d = sc.Dataset()
    d.coords['x'] = sc.scalar(1.0)
    assert len(d.coords.keys()) == 1
    assert 'x' in d.coords.keys()


def test_coords_get():
    d = sc.Dataset()
    d.coords['x'] = sc.scalar(1.0)
    assert sc.identical(d.coords.get('x', sc.scalar(0.0)), d.coords['x'])
    assert sc.identical(d.coords.get('z', d.coords['x']), d.coords['x'])
    assert d.coords.get('z', None) is None
    assert d.coords.get('z') is None


def test_slice_item():
    d = sc.Dataset(
        coords={'x': sc.Variable(dims=['x'], values=np.arange(4, 8))})
    d['a'] = sc.Variable(dims=['x'], values=np.arange(4))
    assert sc.identical(d['a']['x', 2:4].data,
                        sc.Variable(dims=['x'], values=np.arange(2, 4)))
    assert sc.identical(d['a']['x', 2:4].coords['x'],
                        sc.Variable(dims=['x'], values=np.arange(6, 8)))


def test_set_item_slice_from_numpy():
    d = sc.Dataset(
        coords={'x': sc.Variable(dims=['x'], values=np.arange(4, 8))})
    d['a'] = sc.Variable(dims=['x'], values=np.arange(4))
    d['a']['x', 2:4] = np.arange(2)
    assert sc.identical(d['a'].data,
                        sc.Variable(dims=['x'], values=np.array([0, 1, 0, 1])))


def test_set_item_slice_with_variances_from_numpy():
    d = sc.Dataset(
        coords={'x': sc.Variable(dims=['x'], values=np.arange(4, 8))})
    d['a'] = sc.Variable(dims=['x'],
                         values=np.arange(4.0),
                         variances=np.arange(4.0))
    d['a']['x', 2:4].values = np.arange(2)
    d['a']['x', 2:4].variances = np.arange(2, 4)
    assert np.array_equal(d['a'].values, np.array([0.0, 1.0, 0.0, 1.0]))
    assert np.array_equal(d['a'].variances, np.array([0.0, 1.0, 2.0, 3.0]))


def test_iadd_slice():
    d = sc.Dataset(
        coords={'x': sc.Variable(dims=['x'], values=np.arange(4, 8))})
    d['a'] = sc.Variable(dims=['x'], values=np.arange(4))
    d['a']['x', 1] += d['a']['x', 2]
    assert sc.identical(d['a'].data,
                        sc.Variable(dims=['x'], values=np.array([0, 3, 2, 3])))


def test_iadd_range():
    d = sc.Dataset(
        coords={'x': sc.Variable(dims=['x'], values=np.arange(4, 8))})
    d['a'] = sc.Variable(dims=['x'], values=np.arange(4))
    with pytest.raises(RuntimeError):
        d['a']['x', 2:4] += d['a']['x', 1:3]
    d['a']['x', 2:4] += d['a']['x', 2:4]
    assert sc.identical(d['a'].data,
                        sc.Variable(dims=['x'], values=np.array([0, 1, 4, 6])))


def test_contains():
    d = sc.Dataset()
    assert 'a' not in d
    d['a'] = sc.scalar(1.0)
    assert 'a' in d
    assert 'b' not in d
    d['b'] = sc.scalar(1.0)
    assert 'a' in d
    assert 'b' in d


def test_slice():
    d = sc.Dataset(
        data={
            'a': sc.Variable(dims=['x'], values=np.arange(10.0)),
            'b': sc.scalar(1.0)
        },
        coords={'x': sc.Variable(dims=['x'], values=np.arange(10.0))})
    expected = sc.Dataset(
        data={
            'a': sc.DataArray(1.0 * sc.units.one,
                              attrs={'x': 1.0 * sc.units.one}),
            'b': sc.scalar(1.0)
        })
    assert sc.identical(d['x', 1], expected)


def test_chained_slicing():
    x = sc.Variable(dims=['x'], values=np.arange(11.0))
    y = sc.Variable(dims=['y'], values=np.arange(11.0))
    z = sc.Variable(dims=['z'], values=np.arange(11.0))
    d = sc.Dataset(data={
        'a':
        sc.Variable(dims=['z', 'y', 'x'],
                    values=np.arange(1000.0).reshape(10, 10, 10)),
        'b':
        sc.Variable(dims=['x', 'z'],
                    values=np.arange(0.0, 10.0, 0.1).reshape(10, 10))
    },
                   coords={
                       'x': x,
                       'y': y,
                       'z': z
                   })

    expected = sc.Dataset()
    expected['a'] = sc.Variable(dims=['y'],
                                values=np.arange(501.0, 600.0, 10.0))
    expected['b'] = sc.scalar(1.5)
    expected['a'].attrs['x'] = x['x', 1:3]
    expected['b'].attrs['x'] = x['x', 1:3]
    expected['a'].attrs['z'] = z['z', 5:7]
    expected['b'].attrs['z'] = z['z', 5:7]
    expected.coords['y'] = sc.Variable(dims=['y'], values=np.arange(11.0))

    assert sc.identical(d['x', 1]['z', 5], expected)


def test_coords_view_comparison_operators():
    d = sc.Dataset(
        data={
            'a': sc.Variable(dims=['x'], values=np.arange(10.0)),
            'b': sc.scalar(1.0)
        },
        coords={'x': sc.Variable(dims=['x'], values=np.arange(10.0))})

    d1 = sc.Dataset(
        data={
            'a': sc.Variable(dims=['x'], values=np.arange(10.0)),
            'b': sc.scalar(1.0)
        },
        coords={'x': sc.Variable(dims=['x'], values=np.arange(10.0))})
    assert d1['a'].coords == d['a'].coords


def test_sum_mean():
    d = sc.Dataset(data={
        'a':
        sc.Variable(dims=['x', 'y'],
                    values=np.arange(6, dtype=np.int64).reshape(2, 3)),
        'b':
        sc.Variable(dims=['y'], values=np.arange(3, dtype=np.int64))
    },
                   coords={
                       'x':
                       sc.Variable(dims=['x'],
                                   values=np.arange(2, dtype=np.int64)),
                       'y':
                       sc.Variable(dims=['y'],
                                   values=np.arange(3, dtype=np.int64)),
                       'l1':
                       sc.Variable(dims=['x', 'y'],
                                   values=np.arange(6, dtype=np.int64).reshape(
                                       2, 3)),
                       'l2':
                       sc.Variable(dims=['x'],
                                   values=np.arange(2, dtype=np.int64))
                   })
    d_ref = sc.Dataset(data={
        'a':
        sc.Variable(dims=['x'], values=np.array([3, 12], dtype=np.int64)),
        'b':
        sc.scalar(3)
    },
                       coords={
                           'x':
                           sc.Variable(dims=['x'],
                                       values=np.arange(2, dtype=np.int64)),
                           'l2':
                           sc.Variable(dims=['x'],
                                       values=np.arange(2, dtype=np.int64))
                       })

    assert sc.identical(sc.sum(d, 'y'), d_ref)
    assert (sc.mean(d, 'y')['a'].values == [1.0, 4.0]).all()
    assert sc.mean(d, 'y')['b'].value == 1.0


def test_sum_masked():
    d = sc.Dataset(
        data={
            'a':
            sc.Variable(dims=['x'],
                        values=np.array([1, 5, 4, 5, 1], dtype=np.int64))
        })
    d['a'].masks['m1'] = sc.Variable(dims=['x'],
                                     values=np.array(
                                         [False, True, False, True, False]))

    d_ref = sc.Dataset(data={'a': sc.scalar(np.int64(6))})

    result = sc.sum(d, 'x')['a']
    assert sc.identical(result, d_ref['a'])


def test_sum_all():
    da = sc.DataArray(
        sc.Variable(dims=['x', 'y'], values=np.ones(10).reshape(5, 2)))
    ds = sc.Dataset(data={'a': da})
    assert sc.identical(sc.sum(da).data, sc.scalar(10.0))
    assert sc.identical(sc.sum(da), sc.sum(ds)['a'])


def test_nansum_masked():
    d = sc.Dataset(
        data={
            'a':
            sc.Variable(dims=['x'],
                        values=np.array([1, 5, np.nan, np.nan, 1],
                                        dtype=np.float64))
        })
    d['a'].masks['m1'] = sc.Variable(dims=['x'],
                                     values=np.array(
                                         [False, True, False, True, False]))

    d_ref = sc.Dataset(data={'a': sc.scalar(np.float64(2))})

    result = sc.nansum(d, 'x')['a']
    assert sc.identical(result, d_ref['a'])


def test_nansum_all():
    da = sc.DataArray(
        sc.Variable(dims=['x', 'y'], values=np.ones(10).reshape(5, 2)))
    da.data.values[0, 0] = np.nan
    ds = sc.Dataset(data={'a': da})
    assert np.isnan(sc.sum(da).data.value)  # sanity check
    assert sc.identical(sc.nansum(da).data, sc.scalar(9.0))
    assert sc.identical(sc.nansum(da), sc.nansum(ds)['a'])


def test_mean_masked():
    d = sc.Dataset(
        data={
            'a':
            sc.Variable(dims=['x'],
                        values=np.array([1, 5, 4, 5, 1]),
                        dtype=sc.dtype.float64)
        })
    d['a'].masks['m1'] = sc.Variable(dims=['x'],
                                     values=np.array(
                                         [False, True, False, True, False]))
    d_ref = sc.Dataset(data={'a': sc.scalar(2.0)})
    assert sc.identical(sc.mean(d, 'x')['a'], d_ref['a'])
    assert sc.identical(sc.nanmean(d, 'x')['a'], d_ref['a'])


def test_mean_all():
    var = sc.Variable(dims=['x', 'y'], values=np.arange(4.0).reshape(2, 2))
    mask = sc.Variable(dims=['x', 'y'],
                       values=np.array([[False, False], [True, False]]))
    da = sc.DataArray(var, masks={'m': mask})  # Add masks
    assert sc.sum(da).data.value == 0 + 1 + 3  # 2.0 masked
    sc.mean(da).data.value == 4 / 3


def test_nanmean_all():
    var = sc.Variable(dims=['x', 'y'], values=np.arange(4.0).reshape(2, 2))
    var['x', 0]['y', 1].value = np.nan
    mask = sc.Variable(dims=['x', 'y'],
                       values=np.array([[False, False], [True, False]]))
    da = sc.DataArray(var, masks={'m': mask})  # Add masks
    assert sc.nansum(da).data.value == 0 + 3  # 2.0 masked, 1.0 is nan
    sc.mean(da).data.value == 3 / 2


def test_dataset_merge():
    a = sc.Dataset(
        data={'d1': sc.Variable(dims=['x'], values=np.array([1, 2, 3]))})
    b = sc.Dataset(
        data={'d2': sc.Variable(dims=['x'], values=np.array([4, 5, 6]))})
    c = sc.merge(a, b)
    assert sc.identical(a['d1'], c['d1'])
    assert sc.identical(b['d2'], c['d2'])


def test_dataset_concatenate():
    a = sc.Dataset(
        data={'data': sc.Variable(dims=['x'], values=np.array([11, 12]))},
        coords={'x': sc.Variable(dims=['x'], values=np.array([1, 2]))})
    b = sc.Dataset(
        data={'data': sc.Variable(dims=['x'], values=np.array([13, 14]))},
        coords={'x': sc.Variable(dims=['x'], values=np.array([3, 4]))})

    c = sc.concatenate(a, b, 'x')

    assert np.array_equal(c.coords['x'].values, np.array([1, 2, 3, 4]))
    assert np.array_equal(c['data'].values, np.array([11, 12, 13, 14]))


def test_dataset_set_data():
    d1 = sc.Dataset(data={
        'a':
        sc.Variable(dims=['x', 'y'], values=np.random.rand(2, 3)),
        'b':
        sc.scalar(1.0)
    },
                    coords={
                        'x':
                        sc.Variable(dims=['x'],
                                    values=np.arange(2.0),
                                    unit=sc.units.m),
                        'y':
                        sc.Variable(dims=['y'],
                                    values=np.arange(3.0),
                                    unit=sc.units.m),
                        'aux':
                        sc.Variable(dims=['y'], values=np.arange(3))
                    })

    d2 = sc.Dataset(data={
        'a':
        sc.Variable(dims=['x', 'y'], values=np.random.rand(2, 3)),
        'b':
        sc.scalar(1.0)
    },
                    coords={
                        'x':
                        sc.Variable(dims=['x'],
                                    values=np.arange(2.0),
                                    unit=sc.units.m),
                        'y':
                        sc.Variable(dims=['y'],
                                    values=np.arange(3.0),
                                    unit=sc.units.m),
                        'aux':
                        sc.Variable(dims=['y'], values=np.arange(3))
                    })

    d3 = sc.Dataset()
    d3['b'] = d1['a']
    assert sc.identical(d3['b'].data, d1['a'].data)
    assert d3['b'].coords == d1['a'].coords
    d1['a'] = d2['a']
    d1['c'] = d2['a']
    assert sc.identical(d2['a'].data, d1['a'].data)
    assert sc.identical(d2['a'].data, d1['c'].data)

    d = sc.Dataset()
    d['a'] = sc.Variable(dims=['row'],
                         values=np.arange(10.0),
                         variances=np.arange(10.0))
    d['b'] = sc.Variable(dims=['row'], values=np.arange(10.0, 20.0))
    d.coords['row'] = sc.Variable(dims=['row'], values=np.arange(10.0))
    d1 = d['row', 0:1]
    d2 = sc.Dataset(data={'a': d1['a'].data},
                    coords={'row': d1['a'].coords['row']})
    d2['b'] = d1['b']
    expected = sc.Dataset()
    expected['a'] = sc.Variable(dims=['row'],
                                values=np.arange(1.0),
                                variances=np.arange(1.0))
    expected['b'] = sc.Variable(dims=['row'], values=np.arange(10.0, 11.0))
    expected.coords['row'] = sc.Variable(dims=['row'], values=np.arange(1.0))
    assert sc.identical(d2, expected)


def test_binary_with_broadcast():
    d = sc.Dataset(
        data={'data': sc.Variable(dims=['x'], values=np.arange(10.0))},
        coords={'x': sc.Variable(dims=['x'], values=np.arange(10.0))})

    d2 = d - d['x', 0]
    d -= d['x', 0]
    assert sc.identical(d, d2)


def test_binary__with_dataarray():
    da = sc.DataArray(
        data=sc.Variable(dims=['x'], values=np.arange(1.0, 10.0)),
        coords={'x': sc.Variable(dims=['x'], values=np.arange(1.0, 10.0))})
    ds = sc.Dataset(data={da.name: da.copy()})
    orig = ds.copy()
    ds += da
    ds -= da
    ds *= da
    ds /= da
    assert sc.identical(ds, orig)


def test_binary_of_item_with_variable():
    d = sc.Dataset(
        data={'data': sc.Variable(dims=['x'], values=np.arange(10.0))},
        coords={'x': sc.Variable(dims=['x'], values=np.arange(10.0))})
    copy = d.copy()

    d['data'] += 2.0 * sc.units.dimensionless
    d['data'] *= 2.0 * sc.units.m
    d['data'] -= 4.0 * sc.units.m
    d['data'] /= 2.0 * sc.units.m
    assert sc.identical(d, copy)


def test_in_place_binary_with_scalar():
    d = sc.Dataset(data={'data': sc.Variable(dims=['x'], values=[10.0])},
                   coords={'x': sc.Variable(dims=['x'], values=[10])})
    copy = d.copy()

    d += 2
    d *= 2
    d -= 4
    d /= 2
    assert sc.identical(d, copy)


def test_view_in_place_binary_with_scalar():
    d = sc.Dataset(data={'data': sc.Variable(dims=['x'], values=[10.0])},
                   coords={'x': sc.Variable(dims=['x'], values=[10])})
    copy = d.copy()

    d['data'] += 2
    d['data'] *= 2
    d['data'] -= 4
    d['data'] /= 2
    assert sc.identical(d, copy)


def test_add_sum_of_columns():
    d = sc.Dataset(
        data={
            'a': sc.Variable(dims=['x'], values=np.arange(10.0)),
            'b': sc.Variable(dims=['x'], values=np.ones(10))
        })
    d['sum'] = d['a'] + d['b']
    d['a'] += d['b']
    assert sc.identical(d['sum'], d['a'])


def test_name():
    d = sc.Dataset(data={
        'a': sc.Variable(dims=['x'], values=np.arange(10.0)),
        'b': sc.Variable(dims=['x'], values=np.ones(10))
    },
                   coords={'x': sc.Variable(dims=['x'], values=np.arange(10))})
    assert d['a'].name == 'a'
    assert d['b'].name == 'b'
    assert d['x', 2]['b'].name == 'b'
    assert d['b']['x', 2].name == 'b'
    array = d['a'] + d['b']
    assert array.name == ''


def make_simple_dataset(dim1='x', dim2='y', seed=None):
    if seed is not None:
        np.random.seed(seed)
    return sc.Dataset(data={
        'a':
        sc.Variable(dims=[dim1, dim2], values=np.random.rand(2, 3)),
        'b':
        sc.scalar(1.0)
    },
                      coords={
                          dim1:
                          sc.Variable(dims=[dim1],
                                      values=np.arange(2.0),
                                      unit=sc.units.m),
                          dim2:
                          sc.Variable(dims=[dim2],
                                      values=np.arange(3.0),
                                      unit=sc.units.m),
                          'aux':
                          sc.Variable(dims=[dim2], values=np.random.rand(3))
                      })


def test_dataset_view_set_variance():
    d = make_simple_dataset()
    variances = np.arange(6).reshape(2, 3)
    assert d['a'].variances is None
    d['a'].variances = variances
    assert d['a'].variances is not None
    np.testing.assert_array_equal(d['a'].variances, variances)
    d['a'].variances = None
    assert d['a'].variances is None


def test_sort():
    d = sc.Dataset(data={
        'a':
        sc.Variable(dims=['x', 'y'], values=np.arange(6).reshape(2, 3)),
        'b':
        sc.Variable(dims=['x'], values=['b', 'a'])
    },
                   coords={
                       'x':
                       sc.Variable(dims=['x'],
                                   values=np.arange(2.0),
                                   unit=sc.units.m),
                       'y':
                       sc.Variable(dims=['y'],
                                   values=np.arange(3.0),
                                   unit=sc.units.m),
                       'aux':
                       sc.Variable(dims=['x'], values=np.arange(2.0))
                   })
    expected = sc.Dataset(data={
        'a':
        sc.Variable(dims=['x', 'y'],
                    values=np.flip(np.arange(6).reshape(2, 3), axis=0)),
        'b':
        sc.Variable(dims=['x'], values=['a', 'b'])
    },
                          coords={
                              'x':
                              sc.Variable(dims=['x'],
                                          values=[1.0, 0.0],
                                          unit=sc.units.m),
                              'y':
                              sc.Variable(dims=['y'],
                                          values=np.arange(3.0),
                                          unit=sc.units.m),
                              'aux':
                              sc.Variable(dims=['x'], values=[1.0, 0.0])
                          })
    assert sc.identical(sc.sort(d, d['b'].data), expected)


def test_rename_dims():
    d = make_simple_dataset('x', 'y', seed=0)
    original = d.copy()
    renamed = d.rename_dims({'y': 'z'})
    assert sc.identical(d, original)
    renamed.coords['z'] = renamed.coords['y']
    del renamed.coords['y']
    assert sc.identical(renamed, make_simple_dataset('x', 'z', seed=0))
    renamed = renamed.rename_dims(dims_dict={'x': 'y', 'z': 'x'})
    renamed.coords['y'] = renamed.coords['x']
    renamed.coords['x'] = renamed.coords['z']
    del renamed.coords['z']
    assert sc.identical(renamed, make_simple_dataset('y', 'x', seed=0))


def test_coord_delitem():
    var = sc.Variable(dims=['x'], values=np.arange(4))
    d = sc.Dataset(data={'a': var}, coords={'x': var})
    dref = d.copy()
    d.coords['y'] = sc.scalar(1.0)
    assert dref != d
    del d.coords['y']
    assert sc.identical(dref, d)


def test_coords_delitem():
    var = sc.Variable(dims=['x'], values=np.arange(4))
    d = sc.Dataset(data={'a': var}, coords={'x': var})
    dref = d.copy()
    dref.coords['x'] = sc.Variable(dims=['x'], values=np.arange(1, 5))
    assert not sc.identical(d, dref)
    del dref.coords['x']
    assert not sc.identical(d, dref)
    dref.coords['x'] = d.coords['x']
    assert sc.identical(d, dref)


def test_masks_delitem():
    var = sc.Variable(dims=['x'], values=np.array([True, True, False]))
    d = sc.Dataset(data={'a': var}, coords={'x': var})
    dref = d.copy()
    d['a'].masks['masks'] = var
    assert not sc.identical(d, dref)
    del d['a'].masks['masks']
    assert sc.identical(d, dref)


def test_replace():
    v1 = sc.Variable(dims=['x'], values=np.array([1, 2, 3]))
    d = sc.Dataset(data={'a': v1})
    assert sc.identical(d['a'].data, v1)
    v2 = sc.Variable(dims=['x'], values=np.array([4, 5, 6]))
    d['a'].data = v2
    assert sc.identical(d['a'].data, v2)


def test_rebin():
    dataset = sc.Dataset()
    dataset['data'] = sc.Variable(dims=['x'],
                                  values=np.array(10 * [1.0]),
                                  unit=sc.units.counts)
    dataset.coords['x'] = sc.Variable(dims=['x'], values=np.arange(11.0))
    new_coord = sc.Variable(dims=['x'], values=np.arange(0.0, 11, 2))
    dataset = sc.rebin(dataset, 'x', new_coord)
    np.testing.assert_array_equal(dataset['data'].values, np.array(5 * [2]))
    np.testing.assert_array_equal(dataset.coords['x'].values,
                                  np.arange(0, 11, 2))


def _is_copy_of(orig, copy):
    assert sc.identical(orig, copy)
    assert not id(orig) == id(copy)
    orig += 1
    assert sc.identical(orig, copy)


def _is_deep_copy_of(orig, copy):
    assert sc.identical(orig, copy)
    assert not id(orig) == id(copy)
    orig += 1
    assert not sc.identical(orig, copy)


def test_copy():
    import copy
    a = sc.Dataset()
    a['x'] = sc.scalar(1)
    _is_copy_of(a, a.copy(deep=False))
    _is_deep_copy_of(a, a.copy())
    _is_copy_of(a, copy.copy(a))
    _is_deep_copy_of(a, copy.deepcopy(a))


def test_correct_temporaries():
    N = 6
    M = 4
    d1 = sc.Dataset()
    d1['x'] = sc.Variable(dims=['x'], values=np.arange(N).astype(np.float64))
    d1['y'] = sc.Variable(dims=['y'], values=np.arange(M).astype(np.float64))
    arr1 = np.arange(N * M).reshape(N, M).astype(np.float64) + 1
    d1['A'] = sc.Variable(dims=['x', 'y'], values=arr1)
    d1 = d1['x', 1:2]
    assert d1['A'].values.tolist() == [[5.0, 6.0, 7.0, 8.0]]
    d1 = d1['y', 2:3]
    assert list(d1['A'].values) == [7]


def test_iteration():
    var = sc.scalar(1)
    d = sc.Dataset(data={'a': var, 'b': var})
    expected = ['a', 'b']
    for k in d:
        assert k in expected
