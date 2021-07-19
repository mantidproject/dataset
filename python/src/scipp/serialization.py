# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from . import Variable, DataArray


def _encode_dict(meta):
    return {key: encode(var) for key, var in meta.items()}


def encode(obj):
    import msgpack_numpy as m
    if isinstance(obj, Variable):
        enc = {
            '__scipp.Variable__': True,
            'dims': obj.dims,
            'unit': str(obj.unit),
            'values': m.encode(obj.values)
        }
        if obj.variances is not None:
            enc['variances'] = m.encode(obj.variances)
        return enc
    elif isinstance(obj, DataArray):
        return {
            '__scipp.DataArray__': True,
            'data': encode(obj.data),
            'coords': _encode_dict(obj.coords),
            'masks': _encode_dict(obj.masks),
            'attrs': _encode_dict(obj.attrs)
        }
    return obj


def decode(obj):
    import msgpack_numpy as m
    if '__scipp.Variable__' in obj:
        variances = m.decode(obj['variances']) if 'variances' in obj else None
        return Variable(dims=obj['dims'],
                        unit=obj['unit'],
                        values=m.decode(obj['values']),
                        variances=variances)
    if '__scipp.DataArray__' in obj:
        return DataArray(data=obj['data'],
                         coords=obj['coords'],
                         masks=obj['masks'],
                         attrs=obj['attrs'])
    return obj
