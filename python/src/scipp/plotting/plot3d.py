# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

from .objects import make_params, make_plot
from .panel3d import PlotPanel3d
from .view3d import PlotView3d
from .figure3d import PlotFigure3d
from .._shape import flatten


def plot3d(scipp_obj_dict, positions, **kwargs):
    """
    Plot a 3D point cloud through a N dimensional dataset.
    For every dimension above 3, a slider is created to adjust the position of
    the slice in that particular dimension.
    It is possible to add cut surfaces as cartesian, cylindrical or spherical
    planes.
    """
    def builder(*,
                dims,
                norm=None,
                masks=None,
                ax=None,
                pax=None,
                figsize=None,
                cmap=None,
                vmin=None,
                vmax=None,
                title=None,
                background="#f0f0f0",
                pixel_size=None,
                tick_size=None,
                show_outline=True,
                xlabel=None,
                ylabel=None,
                zlabel=None):
        out = {'view_ndims': 1, 'view': PlotView3d}
        params = make_params(cmap=cmap,
                             norm=norm,
                             vmin=vmin,
                             vmax=vmax,
                             masks=masks)
        out['vmin'] = params["values"]["vmin"]
        out['vmax'] = params["values"]["vmax"]
        if len(dims) > 2:
            params['extend_cmap'] = 'both'
        # TODO enable panel once refactored
        #out['panel'] = PlotPanel3d(data_names=list(scipp_obj_dict.keys()))
        out['figure'] = PlotFigure3d(background=background,
                                     cmap=params["values"]["cmap"],
                                     extend=params['extend_cmap'],
                                     figsize=figsize,
                                     mask_cmap=params['masks']['cmap'],
                                     nan_color=params["values"]["nan_color"],
                                     norm=params["values"]["norm"],
                                     pixel_size=pixel_size,
                                     show_outline=show_outline,
                                     tick_size=tick_size,
                                     positions=positions,
                                     xlabel=xlabel,
                                     ylabel=ylabel,
                                     zlabel=zlabel)

        return out

    # TODO if positions not given, make fake coords first
    # Use subclass of PlotModel1d
    # - PlotModel makes fake coords, then flatten in init
    if isinstance(positions, str):
        pos = next(iter(scipp_obj_dict.values())).coords[positions]
    else:
        pos = positions
    print(pos)
    #scipp_obj_dict = {
    #    key: flatten(array, dims=pos.dims, to=''.join(array.dims))
    #    for key, array in scipp_obj_dict.items()
    #}

    return make_plot(builder, scipp_obj_dict, **kwargs)
