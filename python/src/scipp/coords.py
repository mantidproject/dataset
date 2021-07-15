# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock

import inspect
from typing import Union
from . import Variable, DataArray, Dataset


def _add_coords(*, name, obj, tree):
    func = tree[name]
    args = inspect.getfullargspec(func).kwonlyargs
    out = func(**{arg: _get_coord(arg, obj, tree) for arg in args})
    if isinstance(out, Variable):
        obj.coords[name] = out
    else:
        for key, coord in out.items():
            obj.coords[key] = coord


def _get_coord(name, obj, tree):
    if name in obj.meta:
        # Consume coord => switch to attr
        if name in obj.coords:
            obj.attrs[name] = obj.coords[name]
            del obj.coords[name]
        return obj.attrs[name]
    elif isinstance(tree[name], str):
        return _get_coord(tree[name], obj, tree)
    else:
        _add_coords(name=name, obj=obj, tree=tree)
        return _get_coord(name, obj, tree)


def transform_coords(obj: Union[DataArray, Dataset], coords,
                     tree: dict) -> Union[DataArray, Dataset]:
    obj = obj.copy(deep=False)
    for name in coords:
        _add_coords(name=name, obj=obj, tree=tree)
    return obj
