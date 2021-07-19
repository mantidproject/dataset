# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
from ._scipp import core as _cpp
from . import Variable
from ._cpp_wrapper_util import call_func as _call_cpp_func


def convolve(x: Variable, kernel: Variable) -> Variable:
    """Multidimensional convolution.

    The array is convolved with the given kernel.
    Currently this does not handle boundaries and the output is thus smaller
    than the input.

    :param x: Input data.
    :param kernel: kernel.
    """
    return _call_cpp_func(_cpp.convolve, x=x, kernel=kernel)
