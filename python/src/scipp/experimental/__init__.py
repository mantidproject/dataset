# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @file
# @author Simon Heybrock

# flake8: noqa
"""
Staging area for experimental and untested scipp features.

Contents of this submodule are subject to changes and removal without
notice.
"""


def transform_kernel(dtype='float64'):
    if dtype != 'float64':
        raise RuntimeError('Only float64 arguments supported at this point')
    else:
        dtype = 'double'
    import numba
    from inspect import signature

    def decorator(function):
        narg = len(signature(function).parameters)
        return numba.cfunc(dtype + '(' + ','.join([dtype] * narg) +
                           ')')(function)

    return decorator


def transform(kernel, *args):
    """Transform one or more variables using a custom kernel.

    Only variables with dtype=float64 are supported.
    Variances are not supported.

    :param kernel: Kernel to use for the transform. Can be a plain function or
                   a function decorated using `@scipp.experimental.transform_kernel()`
    """
    from .._scipp.core import experimental_transform
    if not hasattr(kernel, 'address'):
        kernel = transform_kernel()(kernel)
    return experimental_transform(kernel, *args)
