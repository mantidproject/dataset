# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

from .tools import find_limits, fix_empty_range
from .._utils import value_to_string
from .._scipp import core as sc
import numpy as np


class PlotController:
    """
    Main controller class.

    This handles all communications between the `PlotWidgets`, `PlotView`,
    `PlotModel`, `PlotPanel` and `PlotProfile`.

    It mainly holds information about limits and dimension labels of coordinate
    axes, which is necessary to update figures in a generic way.
    It also keeps a record of which coordinates contain bin-edges and
    bin-centers, as this information is lost once the model has converted
    everything to bin edges to allow for more generic code.

    """
    def __init__(self,
                 axes=None,
                 name=None,
                 vmin=None,
                 vmax=None,
                 norm=None,
                 scale=None,
                 dim_to_shape=None,
                 coord_shapes=None,
                 multid_coord=None,
                 widgets=None,
                 model=None,
                 panel=None,
                 profile=None,
                 view=None):

        self.widgets = widgets
        self.model = model
        self.panel = panel
        self.profile = profile
        self.view = view

        self.axes = axes
        self.name = name
        self.dim_to_shape = dim_to_shape
        self.update_data_lock = False
        self.axparams = {}

        self.profile_axparams = {}

        self.vmin = vmin
        self.vmax = vmax
        self.norm = norm if norm is not None else "linear"

        self.scale = {dim: "linear" for dim in self.axes.values()}
        if scale is not None:
            for dim, item in scale.items():
                self.scale[dim] = item

        # Save the current profile dimension
        self.profile_dim = None
        # Store coordinate min and max limits
        self.xlims = {}
        # Store labels for sliders
        self.coord_labels = {}
        self.coord_units = {}
        # Record which variables are histograms along which dimension
        self.histograms = {}
        # Keep track if a coordinate with more than one dimension is present
        self.multid_coord = multid_coord

        for key in self.model.get_data_names():

            self.xlims[key] = {}
            self.coord_labels[key] = {}
            self.coord_units[key] = {}
            self.histograms[key] = {}

            # Iterate through axes and collect dimensions
            for dim in self.axes.values():

                coord, label, unit = self.model.get_data_coord(name, dim)

                # To allow for 2D coordinates, the histograms are
                # stored as dicts, with one key per dimension of the coordinate
                self.histograms[key][dim] = {}
                for i, d in enumerate(coord.dims):
                    self.histograms[key][dim][d] = self.dim_to_shape[key][
                        d] == coord_shapes[key][dim][i] - 1

                # The limits for each dimension
                self.xlims[key][dim] = find_limits(coord,
                                                   flip=sc.is_sorted(
                                                       coord,
                                                       dim,
                                                       order='descending'))
                # Check if xmin == xmax
                for scale in self.xlims[key][dim]:
                    self.xlims[key][dim][scale] = fix_empty_range(
                        self.xlims[key][dim][scale])

                    self.xlims[key][dim][scale] = sc.Variable(
                        [dim],
                        values=self.xlims[key][dim][scale],
                        unit=coord.unit)

                self.coord_labels[key][dim] = label
                self.coord_units[key][dim] = unit

        self.initialise_widgets(dim_to_shape[self.name])
        self.initialise_view()
        self.initialise_model()
        if self.profile is not None:
            self.initialise_profile()
            self.connect_profile()

        self.connect_widgets()
        self.connect_view()
        if self.panel is not None:
            self.connect_panel()

    def render(self, norm=None):
        """
        Update axes (and data) to render the figure once all components
        have been created.
        """
        self.update_axes()
        self.update_log_axes_buttons()
        self.update_norm_button(norm)

    def get_dim_shape(self, dim, name=None):
        """
        Get dimension shape.
        """
        if name is None:
            name = self.name
        return self.dim_to_shape[name][dim]

    def get_coord_unit(self, dim, name=None):
        """
        Get dimension coordinate unit.
        """
        if name is None:
            name = self.name
        return self.coord_units[name][dim]

    def initialise_widgets(self, dim_to_shape):
        """
        Initialise widget parameters once the `PlotModel`, `PlotView` and
        `PlotController` have been created.
        """
        ranges = {}
        for dim in self.widgets.get_slider_bounds():
            ranges[dim] = self.model.get_slice_coord_bounds(
                self.name, dim, [0, 1])
        self.widgets.initialise(dim_to_shape, ranges,
                                self.coord_units[self.name])

    def initialise_view(self):
        """
        Send axformatter information to the `PlotView`.
        """
        self.view.initialise(
            axformatters={
                dim: self.model.get_axformatter(self.name, dim)
                for dim in self.axes.values()
            })

    def initialise_profile(self):
        """
        Send axformatter information to the `PlotProfile`.
        """
        self.profile.initialise(
            axformatters={
                dim: self.model.get_axformatter(self.name, dim)
                for dim in self.axes.values()
            })

    def initialise_model(self):
        """
        Dummy initialization for `PlotModel`.
        """
        return

    def connect_widgets(self):
        """
        Connect callbacks to the `PlotWidgets` interface.
        """
        self.widgets.connect({
            "toggle_profile_view": self.toggle_profile_view,
            "update_data": self.update_data,
            "toggle_mask": self.toggle_mask,
            "get_dim_shape": self.get_dim_shape,
            "lock_update_data": self.lock_update_data,
            "unlock_update_data": self.unlock_update_data,
            "swap_dimensions": self.swap_dimensions,
            "get_coord_unit": self.get_coord_unit
        })

    def connect_view(self):
        """
        Connect callbacks to the `PlotView` interface.
        """
        view_callbacks = {
            "update_profile": self.update_profile,
            "toggle_hover_visibility": self.toggle_hover_visibility,
            "keep_line": self.keep_line,
            "remove_line": self.remove_line
        }
        figure_callbacks = {
            "rescale_to_data": self.rescale_to_data,
            "transpose": self.transpose,
            "toggle_xaxis_scale": self.toggle_xaxis_scale,
            "toggle_yaxis_scale": self.toggle_yaxis_scale,
            "toggle_norm": self.toggle_norm,
            "home_view": self.home_view,
            "pan_view": self.pan_view,
            "zoom_view": self.zoom_view,
            "save_view": self.save_view
        }
        self.view.connect(view_callbacks=view_callbacks,
                          figure_callbacks=figure_callbacks)

    def connect_panel(self):
        """
        Dummy connect for `PlotPanel`.
        """
        return

    def connect_profile(self):
        """
        Connect callbacks to the `PlotWidgets` interface.
        """
        self.profile.connect()

    def lock_update_data(self):
        """
        When the thickness slider is changed, the range, and possibly the
        value, of the position slider are changed. We therefore temporary lock
        data updates until all slider ranges and values have been updated
        before manually updating the displayed data slice.
        """
        self.update_data_lock = True

    def unlock_update_data(self):
        """
        Release the data update lock.
        """
        self.update_data_lock = False

    def home_view(self, button=None):
        self.update_axes()

    def pan_view(self, button=None):
        self.view.pan_view()

    def zoom_view(self, button=None):
        self.view.zoom_view()

    def save_view(self, button=None):
        self.view.save_view()

    def find_vmin_vmax(self, button=None):
        """
        Find sensible min and max values for the data.
        If the limits were requested by the user: if the rescale button is
        pressed by the user, it means we forcibly update the self.vmin/vmax.
        If no button is pressed, we use the global limits instead of the
        min and max values found by the model.
        """
        [vmin, vmax] = self.model.rescale_to_data(scale=self.norm)
        if self.vmin is not None:
            if button is None:
                vmin = self.vmin
            else:
                self.vmin = None
        if self.vmax is not None:
            if button is None:
                vmax = self.vmax
            else:
                self.vmax = None
        return vmin, vmax

    def rescale_to_data(self, button=None):
        """
        Automatically rescale the y axis (1D plot) or the colorbar (2D+3D
        plots) to the minimum and maximum value inside the currently displayed
        data slice.
        """
        vmin, vmax = self.find_vmin_vmax(button=button)
        self.view.rescale_to_data(vmin, vmax)
        if self.panel is not None:
            self.panel.rescale_to_data(vmin=vmin,
                                       vmax=vmax,
                                       mask_info=self.get_masks_info())
        self.refresh()

    def refresh(self):
        """
        Get current slice values from model and update the view.
        This is used when either vmin and vmax limits have been changed, or
        the normalization has been modified.
        """
        new_values = self.model.get_slice_values(
            mask_info=self.get_masks_info())
        self.view.update_data(new_values)

    def transpose(self, owner=None):
        """
        Transpose the displayed axes.
        """
        xyz = self._get_xyz_axes()
        dims = [self.axes[key] for key in xyz]
        keys = list(np.roll(xyz, 1))
        for i in range(len(dims)):
            self.axes[keys[i]] = dims[i]
        self.update_axes()
        self.update_log_axes_buttons()

    def toggle_xaxis_scale(self, owner, normalize=False):
        """
        Toggle x-axis scale from toolbar button signal.
        """
        dim = self.axes["x"]
        self.scale[dim] = "log" if owner.value else "linear"
        self.update_axes(normalize=normalize)

    def toggle_yaxis_scale(self, owner, normalize=False):
        """
        Toggle y-axis scale from toolbar button signal.
        """
        dim = self.axes["y"]
        self.scale[dim] = "log" if owner.value else "linear"
        self.update_axes(normalize=normalize)

    def toggle_zaxis_scale(self, owner, normalize=False):
        """
        Toggle z-axis scale from toolbar button signal.
        """
        dim = self.axes["z"]
        self.scale[dim] = "log" if owner.value else "linear"
        self.update_axes(normalize=normalize)

    def toggle_norm(self, owner):
        """
        Toggle data normalization from toolbar button signal.
        """
        self.norm = "log" if owner.value else "linear"
        vmin, vmax = self.find_vmin_vmax()
        self.view.toggle_norm(self.norm, vmin, vmax)
        self.refresh()

    def swap_dimensions(self, index, old_dim, new_dim):
        """
        Swap one dimension for another in the displayed axes.
        """
        pos = list(self.axes.values()).index(new_dim)
        key = list(self.axes.keys())[pos]
        self.axes[key] = old_dim
        self.axes[index] = new_dim
        self.update_axes()
        self.update_log_axes_buttons()
        # Update the slider readout here because the widgets do not have access
        # to the model, which holds the coordinates.
        # ranges = {}
        lower, upper = self.model.get_slice_coord_bounds(
            self.name, new_dim, [0, 1])
        self.widgets.update_slider_readout(index, lower, upper, [0, 1],
                                           new_dim == self.multid_coord)

    def update_log_axes_buttons(self):
        """
        When either axes or dimensions are swapped, we need to update the
        status (color and value) of the toolbar log buttons. We send this to
        the view which in turn with forward it to the toolbar.
        """
        self.view.update_log_axes_buttons(
            {ax: self.scale[self.axes[ax]]
             for ax in self._get_xyz_axes()})

    def update_norm_button(self, *args, **kwargs):
        """
        Change state of norm button according to supplied norm value.
        """
        self.view.update_norm_button(*args, **kwargs)

    def update_axes(self, change=None, normalize=True):
        """
        This function is called when a dimension that is displayed along a
        given axis is changed. This happens for instance when we want to
        flip/transpose a 2D image, or display a new dimension along the x-axis
        in a 1D plot.
        This function gathers the relevant parameters about the axes currently
        selected for display, and then offloads the computation of the new
        state to the model. If then gets the updated data back from the model
        and sends it over to the view for display.
        """
        self.axparams = self._make_axes_parameters()
        other_params = self.model.update_axes(self.axparams)
        if other_params is not None:
            self.axparams.update(other_params)
        self.view.update_axes(axparams=self.axparams)
        if self.panel is not None:
            self.panel.update_axes(axparams=self.axparams)
        if self.profile is not None:
            self.toggle_profile_view()
        self.update_data()
        if normalize:
            self.rescale_to_data()

    def update_data(self, change=None):
        """
        This function is called when the data in the displayed 1D plot or 2D
        image is to be updated. This happens for instance when we move a slider
        which is navigating an additional dimension. It is also always
        called when update_axes is called since the displayed data needs to be
        updated when the axes have changed.
        """
        owner_dim = None

        if self.update_data_lock:
            return

        slices = self.widgets.get_slider_bounds()
        if change is not None:
            owner_dim = self.widgets.get_index_dim(change["owner"].index)
            lower, upper = self.model.get_slice_coord_bounds(
                self.name, owner_dim, slices[owner_dim])
            self.widgets.update_slider_readout(change["owner"].index, lower,
                                               upper, slices[owner_dim],
                                               owner_dim == self.multid_coord)

        info = {"slice_label": self._make_slice_label(slices, "")[1:]}

        new_values = self.model.update_data(slices,
                                            mask_info=self.get_masks_info())
        self.view.update_data(new_values, info=info)
        if self.panel is not None:
            self.panel.update_data(info)
        if self.profile_dim is not None:
            if owner_dim == self.profile_dim:
                self.profile.update_slice_area(lower, upper)
            else:
                self.model.update_profile_model(visible=True,
                                                slices=slices,
                                                profile_dim=self.profile_dim)

    def toggle_mask(self, change):
        """
        Hide or show a given mask.
        """
        self.view.toggle_mask(change)
        self.refresh()
        if self.profile is not None:
            self.profile.toggle_mask(change["owner"].mask_group,
                                     change["owner"].mask_name, change["new"])

    def _get_xyz_axes(self):
        """
        Get the list of displated axes for the current plot. This is ["x"] for
        1d plots, ["x", "y"] for 2d plots, and ["x", "y", "z"] for 3d plots.
        """
        return sorted(list(set(['x', 'y', 'z']) & set(self.axes.keys())))

    def _make_axes_parameters(self):
        """
        Gather the information (dimensions, limits, etc...) about the (x, y, z)
        axes that are displayed on the plots.
        """
        axparams = {}
        for ax in self._get_xyz_axes():
            dim = self.axes[ax]
            xmin = np.Inf
            xmax = np.NINF
            for name in self.xlims:
                xlims = self.xlims[name][dim][self.scale[dim]].values
                xmin = min(xmin, xlims[0])
                xmax = max(xmax, xlims[1])
            axparams[ax] = {
                "lims": np.array([xmin, xmax]),
                "scale": self.scale[dim],
                "hist": {
                    name: self.histograms[name][dim][dim]
                    for name in self.histograms
                },
                "dim": dim,
                "label": self.coord_labels[self.name][dim]
            }

        return axparams

    def get_masks_info(self):
        """
        Get information on masks from widgets.
        """
        return self.widgets.get_masks_info()

    def keep_line(self, target=None, name=None, color=None, line_id=None):
        """
        Get a message from either a panel (1d plot) or a view (picking) to keep
        the currently displayed line or profile.
        Send the message to the appropriate target: either the view1d or the
        profile view.
        """
        if name is None:
            name = self.name
        if target == "profile":
            self.profile.keep_line(name=name, color=color, line_id=line_id)
            self.profile.rescale_to_data()
        else:
            self.view.keep_line(name=name, color=color, line_id=line_id)

    def remove_line(self, target=None, name=None, line_id=None):
        """
        Get a message from either a panel (1d plot) or a view (picking) to
        remove a given line or profile.
        Send the message to the appropriate target: either the view1d or the
        profile view.
        """
        if name is None:
            name = self.name
        if target == "profile":
            self.profile.remove_line(name=name, line_id=line_id)
        else:
            self.view.remove_line(name=name, line_id=line_id)

    def toggle_profile_view(self, owner=None):
        """
        Show or hide the 1d plot displaying the profile along an additional
        dimension.
        As we do this, we also collect some information on the limits of the
        view area to be displayed.
        """
        if owner is None:
            self.profile_dim = None
            visible = False
            self.widgets.clear_profile_buttons()
        else:
            self.profile_dim = self.widgets.get_index_dim(owner.index)
            self.profile_axparams.clear()
            if owner.button_style == "info":
                owner.button_style = ""
                visible = False
            else:
                owner.button_style = "info"
                self.widgets.clear_profile_buttons(exclude=owner.index)
                visible = True

            if visible:
                xmin = np.Inf
                xmax = np.NINF
                for name in self.xlims:
                    xlims = self.xlims[name][self.profile_dim][self.scale[
                        self.profile_dim]].values
                    xmin = min(xmin, xlims[0])
                    xmax = max(xmax, xlims[1])
                self.profile_axparams = {
                    "x": {
                        "lims": [xmin, xmax],
                        "scale": self.scale[self.profile_dim],
                        "hist": {
                            name: self.histograms[name][self.profile_dim][
                                self.profile_dim]
                            for name in self.histograms
                        },
                        "dim": self.profile_dim,
                        "label": self.coord_labels[self.name][self.profile_dim]
                    }
                }

                self.profile.update_axes(axparams=self.profile_axparams)
            if not visible or self.profile.is_visible():
                self.view.reset_profile()

        self.profile.toggle_view(visible=visible)
        self.toggle_hover_visibility(False)
        self.view.update_profile_connection(visible=visible)

        if visible:
            slices = self.widgets.get_slider_bounds()
            self.model.update_profile_model(visible=visible,
                                            slices=slices,
                                            profile_dim=self.profile_dim)
            lower, upper = self.model.get_slice_coord_bounds(
                self.name, self.profile_dim, slices[self.profile_dim])
            self.profile.update_slice_area(lower, upper)

    def update_profile(self, xdata=None, ydata=None):
        """
        This is called from a mouse move event, which requires an update of the
        currently displayed profile.
        We gather the information on which dims should be sliced by the model,
        ask the model to slice down the data, and send the new data returned by
        the model to the profile view.
        """
        info = {"slice_label": ""}
        ax_dims = {self.axparams[xyz]["dim"]: xyz for xyz in self.axparams}
        xydata = {'x': xdata, 'y': ydata}

        slices = self.widgets.get_slider_bounds(exclude=self.profile_dim)

        # Add pixel locations to profile label
        for dim in ax_dims:
            info["slice_label"] = "{},{}:{}".format(
                info["slice_label"], dim,
                value_to_string(xydata[ax_dims[dim]], precision=1))

        info["slice_label"] = self._make_slice_label(slices,
                                                     info["slice_label"])[1:]

        # Get new values from model
        new_values = self.model.update_profile(xdata=xdata,
                                               ydata=ydata,
                                               slices=slices,
                                               axparams=self.profile_axparams,
                                               profile_dim=self.profile_dim,
                                               mask_info=self.get_masks_info())
        # Send new values to the profile view
        self.profile.update_data(new_values, info=info)

    def _make_slice_label(self, slices, label):
        # Add slice ranges to profile label
        for dim in slices:
            lower, upper = self.model.get_slice_coord_bounds(
                self.name, dim, slices[dim])
            label = "{},{}[{}]".format(
                label, dim,
                self.widgets.get_slice_extent(lower, upper, slices[dim],
                                              dim == self.multid_coord))
        return label

    def toggle_hover_visibility(self, value):
        """
        Show/hide the profile view depending on the value of the profile button
        in the widgets.
        """
        self.profile.toggle_hover_visibility(value)
