# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @file
import scipp as sc


def test_version():
    assert len(sc.__version__) > 0
