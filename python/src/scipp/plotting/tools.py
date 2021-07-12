# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

from .. import config
from ..utils import name_with_unit
from .._scipp import core as sc
import numpy as np
from copy import copy
import io


def get_line_param(name=None, index=None):
    """
    Get the default line parameter from the config.
    If an index is supplied, return the i-th item in the list.
    """
    param = getattr(config.plot, name)
    return param[index % len(param)]


def to_bin_centers(x, dim):
    """
    Convert array edges to centers
    """
    return 0.5 * (x[dim, 1:] + x[dim, :-1])


def to_bin_edges(x, dim):
    """
    Convert array centers to edges
    """
    idim = x.dims.index(dim)
    if x.shape[idim] < 2:
        one = 1.0 * x.unit
        return sc.concatenate(x[dim, 0:1] - one, x[dim, 0:1] + one, dim)
    else:
        center = to_bin_centers(x, dim)
        # Note: use range of 0:1 to keep dimension dim in the slice to avoid
        # switching round dimension order in concatenate step.
        left = center[dim, 0:1] - (x[dim, 1] - x[dim, 0])
        right = center[dim, -1] + (x[dim, -1] - x[dim, -2])
        return sc.concatenate(sc.concatenate(left, center, dim), right, dim)


def parse_params(params=None,
                 defaults=None,
                 globs=None,
                 variable=None,
                 array=None,
                 name=None):
    """
    Construct the colorbar settings using default and input values
    """
    from matplotlib.colors import Normalize, LogNorm, LinearSegmentedColormap
    from matplotlib import cm

    parsed = dict(config.plot.params)
    if defaults is not None:
        for key, val in defaults.items():
            parsed[key] = val
    if globs is not None:
        for key, val in globs.items():
            # Global parameters need special treatment because by default they
            # are set to None, and we don't want to overwrite the defaults.
            if val is not None:
                parsed[key] = val
    if params is not None:
        if isinstance(params, bool):
            params = {"show": params}
        for key, val in params.items():
            parsed[key] = val

    if parsed["norm"] == "log":
        norm = LogNorm
    elif parsed["norm"] == "linear":
        norm = Normalize
    else:
        raise RuntimeError("Unknown norm. Expected 'linear' or 'log', "
                           "got {}.".format(parsed["norm"]))
    parsed["norm"] = norm(vmin=parsed["vmin"], vmax=parsed["vmax"])

    # Convert color into custom colormap
    if parsed["color"] is not None:
        parsed["cmap"] = LinearSegmentedColormap.from_list(
            "tmp", [parsed["color"], parsed["color"]])
    else:
        parsed["cmap"] = copy(cm.get_cmap(parsed["cmap"]))

    if parsed["under_color"] is None:
        parsed["cmap"].set_under(parsed["cmap"](0.0))
    else:
        parsed["cmap"].set_under(parsed["under_color"])
    if parsed["over_color"] is None:
        parsed["cmap"].set_over(parsed["cmap"](1.0))
    else:
        parsed["cmap"].set_over(parsed["over_color"])

    if variable is not None:
        parsed["unit"] = name_with_unit(var=variable, name="")

    return parsed


def make_fake_coord(dim, size, unit=None):
    """
    Make a Variable with indices as values, to be used as a fake coordinate
    for either missing coordinates or non-number coordinates (e.g. vector or
    string coordinates).
    """
    kwargs = {"values": np.arange(size, dtype=np.float64)}
    if unit is not None:
        kwargs["unit"] = unit
    return sc.Variable(dims=[dim], **kwargs)


def vars_to_err(v):
    """
    Convert variances to errors.
    """
    with np.errstate(invalid="ignore"):
        v = np.sqrt(v)
    np.nan_to_num(v, copy=False)
    return v


def find_log_limits(x):
    """
    To find log scale limits, we histogram the data between 1.0-30
    and 1.0e+30 and include only bins that are non-zero.
    """
    from .. import flatten, ones
    volume = np.product(x.shape)
    pixel = flatten(sc.values(x), dim='pixel')
    weights = ones(dims=['pixel'], shape=[volume])
    hist = sc.histogram(sc.DataArray(data=weights, coords={'order': pixel}),
                        bins=sc.Variable(dims=['order'],
                                         values=np.geomspace(1e-30,
                                                             1e30,
                                                             num=61),
                                         unit=x.unit))
    # Find the first and the last non-zero bins
    inds = np.nonzero((hist.data > 0.0 * sc.units.one).values)
    ar = np.arange(hist.data.shape[0])[inds]
    # Safety check in case there are no values in range 1.0e-30:1.0e+30:
    # fall back to the linear method and replace with arbitrary values if the
    # limits are negative.
    if len(ar) == 0:
        [vmin, vmax] = find_linear_limits(x)
        if vmin <= 0.0:
            if vmax <= 0.0:
                vmin = 0.1
                vmax = 1.0
            else:
                vmin = 1.0e-3 * vmax
    else:
        vmin = hist.coords['order']['order', ar.min()].value
        vmax = hist.coords['order']['order', ar.max() + 1].value
    return [vmin, vmax]


def find_linear_limits(x):
    """
    Find variable min and max.
    """
    return [sc.nanmin(x).value, sc.nanmax(x).value]


def find_limits(x, scale=None, flip=False):
    """
    Find sensible limits, depending on linear or log scale.
    """
    if scale is not None:
        if scale == "log":
            lims = {"log": find_log_limits(x)}
        else:
            lims = {"linear": find_linear_limits(x)}
    else:
        lims = {"log": find_log_limits(x), "linear": find_linear_limits(x)}
    if flip:
        for key in lims:
            lims[key] = np.flip(lims[key]).copy()
    return lims


def fix_empty_range(lims, replacement=None):
    """
    Range correction in case xmin == xmax
    """
    dx = 0.0
    if lims[0] == lims[1]:
        if replacement is not None:
            dx = 0.5 * replacement
        elif lims[0] == 0.0:
            dx = 0.5
        else:
            dx = 0.5 * abs(lims[0])
    return [lims[0] - dx, lims[1] + dx]


def fig_to_pngbytes(fig):
    """
    Convert figure to png image bytes.
    We also close the figure to prevent it from showing up again in
    cells further down the notebook.
    """
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def to_dict(meta):
    """
    Convert a coords, meta, attrs or masks object to a python dict.
    """
    return {name: var for name, var in meta.items()}
