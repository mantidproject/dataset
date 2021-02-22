# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

from .. import config
from .controller3d import PlotController3d
from .model3d import PlotModel3d
from .panel3d import PlotPanel3d
from .sciplot import SciPlot
from .view3d import PlotView3d
from .widgets import PlotWidgets


def plot3d(*args, filename=None, **kwargs):
    """
    Plot a 3D point cloud through a N dimensional dataset.
    For every dimension above 3, a slider is created to adjust the position of
    the slice in that particular dimension.
    It is possible to add cut surfaces as cartesian, cylindrical or spherical
    planes.
    """

    # In 3d scenes, the size of the pixels depends on the display's pixel
    # scaling ratio, so we retrieve this from a javascript variable run when
    # we imported the plot module.
    if "pixel_ratio" not in config.plot:
        try:
            from IPython import get_ipython
            ipy = get_ipython()
            if ipy is not None:
                pixel_ratio = ipy.kernel.shell.user_ns.get("devicePixelRatio")
                # Note that pixel_ratio appears to be None when building the
                # documentation, possibly because the page is rendered in one
                # go, before the asynchronous javascript has been run.
                # See https://stackoverflow.com/questions/30902898
                # So we do not update the config if it is None, and it will
                # default to 1.0 in figure3d.py.
                if pixel_ratio is not None:
                    config.update({'plot.pixel_ratio': pixel_ratio})
        except ImportError:
            pass

    sp = SciPlot3d(*args, **kwargs)
    if filename is not None:
        sp.savefig(filename)
    else:
        return sp


class SciPlot3d(SciPlot):
    """
    Class for 3 dimensional plots.

    It uses `pythreejs` to render the data as an interactive 3d point cloud.
    All points are rendered as 2d squares for best performance.
    It also holds a `panel3d` with additional control for spatial slicing based
    on altering the opacity of the points.
    """
    def __init__(self,
                 scipp_obj_dict=None,
                 positions=None,
                 axes=None,
                 figsize=None,
                 aspect=None,
                 masks=None,
                 cmap=None,
                 norm=None,
                 scale=None,
                 vmin=None,
                 vmax=None,
                 color=None,
                 background="#f0f0f0",
                 pixel_size=None,
                 tick_size=None,
                 show_outline=True,
                 xlabel=None,
                 ylabel=None,
                 zlabel=None):

        view_ndims = 3

        scipp_obj_dict = {
            key: array if array.bins is None else array.bins.sum()
            for key, array in scipp_obj_dict.items()
        }

        super().__init__(scipp_obj_dict=scipp_obj_dict,
                         axes=axes,
                         cmap=cmap,
                         norm=norm,
                         vmin=vmin,
                         vmax=vmax,
                         masks=masks,
                         positions=positions,
                         view_ndims=view_ndims)

        # The model which takes care of all heavy calculations
        self.model = PlotModel3d(scipp_obj_dict=scipp_obj_dict,
                                 axes=self.axes,
                                 name=self.name,
                                 dim_to_shape=self.dim_to_shape,
                                 dim_label_map=self.dim_label_map,
                                 positions=positions)

        # Run validation checks before rendering the plot.
        # Note that validation needs to be run after model is created.
        self.validate()

        # Create control widgets (sliders and buttons)
        self.widgets = PlotWidgets(axes=self.axes,
                                   ndim=view_ndims,
                                   name=self.name,
                                   dim_to_shape=self.dim_to_shape,
                                   dim_label_map=self.dim_label_map,
                                   masks=self.masks,
                                   pos_dims=self.position_dims,
                                   multid_coord=self.model.get_multid_coord())

        # The view which will display the 3d scene and send pick events back to
        # the controller
        self.view = PlotView3d(
            cmap=self.params["values"][self.name]["cmap"],
            norm=self.params["values"][self.name]["norm"],
            unit=self.params["values"][self.name]["unit"],
            masks=self.masks[self.name],
            nan_color=self.params["values"][self.name]["nan_color"],
            tick_size=tick_size,
            background=background,
            show_outline=show_outline,
            figsize=figsize,
            extend=self.extend_cmap,
            xlabel=xlabel,
            ylabel=ylabel,
            zlabel=zlabel)

        # An additional panel view with widgets to control the cut surface
        self.panel = PlotPanel3d(positions=positions,
                                 unit=self.params["values"][self.name]["unit"])

        # The main controller module which connects all the parts
        self.controller = PlotController3d(
            axes=self.axes,
            aspect=aspect,
            name=self.name,
            dim_to_shape=self.dim_to_shape,
            coord_shapes=self.coord_shapes,
            norm=norm,
            vmin=self.params["values"][self.name]["vmin"],
            vmax=self.params["values"][self.name]["vmax"],
            scale=scale,
            pixel_size=pixel_size,
            positions=positions,
            widgets=self.widgets,
            model=self.model,
            view=self.view,
            panel=self.panel,
            profile=self.profile,
            multid_coord=self.model.get_multid_coord())

        # Render the figure once all components have been created.
        self.render(norm=norm)
