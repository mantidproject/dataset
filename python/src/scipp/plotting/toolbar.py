# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

import ipywidgets as ipw


class PlotToolbar:
    """
    Custom toolbar with additional buttons for controlling log scales and
    normalization, and with back/forward buttons removed.
    """
    def __init__(self, canvas=None):

        # Prepare containers
        self.container = ipw.VBox()
        self.members = {}

        # Keep a reference to the matplotlib toolbar so we can call the zoom
        # and pan methods
        self.mpl_toolbar = None

        if canvas is not None:
            canvas.toolbar_visible = False
            self.mpl_toolbar = canvas.toolbar

        self.add_button(name="home_view",
                        icon="home",
                        tooltip="Reset original view")

    def _ipython_display_(self):
        """
        IPython display representation for Jupyter notebooks.
        """
        return self._to_widget()._ipython_display_()

    def _to_widget(self):
        """
        Return the VBox container
        """
        return self.container

    def set_visible(self, visible):
        """
        Need to hide/show the toolbar when a canvas is hidden/shown.
        """
        self.container.layout.display = None if visible else 'none'

    def add_button(self, name, **kwargs):
        """
        Create a new button and add it to the toolbar members list.
        """
        args = self._parse_button_args(**kwargs)
        self.members[name] = ipw.Button(**args)

    def add_togglebutton(self, name, **kwargs):
        """
        Create a fake ToggleButton using Button because sometimes we want to
        change the value of the button without triggering an update, e.g. when
        we swap the axes.
        """
        args = self._parse_button_args(**kwargs)
        self.members[name] = ipw.Button(**args)
        setattr(self.members[name], "value", False)
        # Add a local observer to change the color of the button according to
        # its value.
        self.members[name].on_click(self.toggle_button_color)

    def toggle_button_color(self, owner, value=None):
        """
        Change the color of the button to make it look like a ToggleButton.
        """
        if value is None:
            owner.value = not owner.value
        else:
            owner.value = value
        owner.style.button_color = "#bdbdbd" if owner.value else "#eeeeee"

    def connect(self, callbacks):
        """
        Connect callbacks to button clicks.
        """
        for key in callbacks:
            if key in self.members:
                self.members[key].on_click(callbacks[key])

    def _update_container(self):
        """
        Update the container's children according to the buttons in the
        members.
        """
        self.container.children = tuple(self.members.values())

    def _parse_button_args(self, layout=None, **kwargs):
        """
        Parse button arguments and add some default styling options.
        """
        args = {"layout": {"width": "34px", "padding": "0px 0px 0px 0px"}}
        if layout is not None:
            args["layout"].update(layout)
        for key, value in kwargs.items():
            if value is not None:
                args[key] = value
        return args

    def update_log_axes_buttons(self, axes_scales):
        """
        When axes are changed or swapped, update the value and color of the
        custom togglebuttons without triggering a new update.
        """
        for xyz, scale in axes_scales.items():
            key = "toggle_{}axis_scale".format(xyz)
            if key in self.members:
                self.toggle_button_color(self.members[key],
                                         value=scale == "log")

    def update_norm_button(self, norm=None):
        """
        Change state of norm button according to supplied norm value.
        """
        self.toggle_button_color(self.members["toggle_norm"],
                                 value=norm == "log")

    def home_view(self):
        self.mpl_toolbar.home()

    def pan_view(self):
        # In case the zoom button is selected, we need to de-select it
        if self.members["zoom_view"].value:
            self.toggle_button_color(self.members["zoom_view"])
        self.mpl_toolbar.pan()

    def zoom_view(self):
        # In case the pan button is selected, we need to de-select it
        if self.members["pan_view"].value:
            self.toggle_button_color(self.members["pan_view"])
        self.mpl_toolbar.zoom()

    def save_view(self):
        self.mpl_toolbar.save_figure()

    def rescale_on_zoom(self):
        return self.members["zoom_view"].value


class PlotToolbar1d(PlotToolbar):
    """
    Custom toolbar for 1d figures.
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.add_togglebutton(name="pan_view", icon="arrows", tooltip="Pan")
        self.add_togglebutton(name="zoom_view",
                              icon="square-o",
                              tooltip="Zoom")
        self.add_button(name="rescale_to_data",
                        icon="arrows-v",
                        tooltip="Rescale")
        self.add_togglebutton(name="toggle_xaxis_scale",
                              description="logx",
                              tooltip="Log(x)")
        self.add_togglebutton(name="toggle_norm",
                              description="logy",
                              tooltip="Log(y)")
        self.add_button(name="save_view", icon="save", tooltip="Save")
        self._update_container()


class PlotToolbar2d(PlotToolbar):
    """
    Custom toolbar for 2d figures.
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.add_togglebutton(name="pan_view", icon="arrows", tooltip="Pan")
        self.add_togglebutton(name="zoom_view",
                              icon="square-o",
                              tooltip="Zoom")
        self.add_button(name="rescale_to_data",
                        icon="arrows-v",
                        tooltip="Rescale")
        self.add_button(name="transpose", icon="retweet", tooltip="Transpose")
        self.add_togglebutton(name="toggle_xaxis_scale",
                              description="logx",
                              tooltip="Log(x)")
        self.add_togglebutton(name="toggle_yaxis_scale",
                              description="logy",
                              tooltip="Log(y)")
        self.add_togglebutton(name="toggle_norm",
                              description="log",
                              tooltip="Log(data)")
        self.add_button(name="save_view", icon="save", tooltip="Save")
        self._update_container()


class PlotToolbar3d(PlotToolbar):
    """
    Custom toolbar for 3d figures.
    """
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.add_button(name="camera_x_normal",
                        icon="camera",
                        description="X",
                        tooltip="Camera to X normal. "
                        "Click twice to flip the view direction.")
        self.add_button(name="camera_y_normal",
                        icon="camera",
                        description="Y",
                        tooltip="Camera to Y normal. "
                        "Click twice to flip the view direction.")
        self.add_button(name="camera_z_normal",
                        icon="camera",
                        description="Z",
                        tooltip="Camera to Z normal. "
                        "Click twice to flip the view direction.")
        self.add_button(name="rescale_to_data",
                        icon="arrows-v",
                        tooltip="Rescale")
        self.add_togglebutton(name="toggle_norm",
                              description="log",
                              tooltip="Log(data)")
        self._update_container()
