# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Neil Vaytet

from .. import config
import ipywidgets as ipw
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import io


class PlotFigure:
    """
    Base class for 1d and 2d figures, that holds matplotlib axes.
    """
    def __init__(self,
                 ax=None,
                 cax=None,
                 figsize=None,
                 title=None,
                 padding=None,
                 ndim=1,
                 xlabel=None,
                 ylabel=None,
                 toolbar=None):
        self.fig = None
        self.ax = ax
        self.cax = cax
        self.own_axes = True
        self.toolbar = None
        if self.ax is None:
            if figsize is None:
                figsize = (config.plot.width / config.plot.dpi,
                           config.plot.height / config.plot.dpi)
            self.fig, self.ax = plt.subplots(1,
                                             1,
                                             figsize=figsize,
                                             dpi=config.plot.dpi)
            if padding is None:
                padding = config.plot.padding
            self.fig.tight_layout(rect=padding)
            if self.is_widget():
                # We create a custom toolbar
                self.toolbar = toolbar(canvas=self.fig.canvas)
                self.fig.canvas.toolbar_visible = False
        else:
            self.own_axes = False
            self.fig = self.ax.get_figure()

        self.ax.set_title(title)

        self.axformatter = {}
        self.axlocator = {}
        self.xlabel = xlabel
        self.ylabel = ylabel

    def is_widget(self):
        """
        Check whether we are using the Matplotlib widget backend or not.
        "on_widget_constructed" is an attribute specific to `ipywidgets`.
        """
        return hasattr(self.fig.canvas, "on_widget_constructed")

    def savefig(self, filename=None):
        """
        Save plot to file.
        Possible file extensions are `.jpg`, `.png` and `.pdf`.
        The default directory for writing the file is the same as the
        directory where the script or notebook is running.
        """
        self.fig.savefig(filename, bbox_inches="tight")

    def _ipython_display_(self):
        """
        IPython display representation for Jupyter notebooks.
        """
        return self._to_widget()._ipython_display_()

    def _to_widget(self, as_static=False):
        """
        Convert the Matplotlib figure to a widget. If the ipympl (widget)
        backend is in use, return the custom toolbar and the figure canvas.
        If not, convert the plot to a png image and place inside an ipywidgets
        Image container.
        """
        if self.is_widget():
            if as_static:
                return ipw.HBox([self.toolbar._to_widget(), self._to_image()])
            else:
                return ipw.HBox([self.toolbar._to_widget(), self.fig.canvas])
        else:
            return self._to_image()

    def _to_image(self):
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png')
        # Here we close the figure to prevent it from showing up again in
        # cells further down the notebook.
        plt.close(self.fig)
        buf.seek(0)
        return ipw.Image(value=buf.getvalue(),
                         width=config.plot.width,
                         height=config.plot.height)

    def show(self):
        """
        Show the matplotlib figure.
        """
        self.fig.show()

    def initialise(self, axformatters=None):
        """
        Initialise figure parameters once the model has been created, since
        the axes formatters are defined by the model.
        """
        for dim in axformatters:
            self.axformatter[dim] = {}
            for key in ["linear", "log"]:
                if axformatters[dim][key] is None:
                    self.axformatter[dim][key] = ticker.ScalarFormatter()
                else:
                    self.axformatter[dim][key] = ticker.FuncFormatter(
                        axformatters[dim][key])
            self.axlocator[dim] = {
                "linear": ticker.AutoLocator(),
                "log": ticker.LogLocator()
            }
            if axformatters[dim]["custom_locator"]:
                self.axlocator[dim]["linear"] = ticker.MaxNLocator(
                    integer=True)

    def connect(self, callbacks):
        """
        Connect the toolbar to callback from the controller. This includes
        rescaling the data norm, and change the scale (log or linear) on the
        axes.
        """
        if self.toolbar is not None:
            self.toolbar.connect(callbacks)

    def draw(self):
        """
        Manually update the figure.
        We control update manually since we have better control on how many
        draws are performed on the canvas. Matplotlib's automatic drawing
        (which we have disabled by using `plt.ioff()`) can degrade performance
        significantly.
        """
        self.fig.canvas.draw_idle()

    def connect_profile(self, pick_callback=None, hover_callback=None):
        """
        Connect the figure to provided callbacks to handle profile picking and
        hovering updates.
        """
        pick_connection = self.fig.canvas.mpl_connect('pick_event',
                                                      pick_callback)
        hover_connection = self.fig.canvas.mpl_connect('motion_notify_event',
                                                       hover_callback)
        return pick_connection, hover_connection

    def disconnect_profile(self, pick_connection=None, hover_connection=None):
        """
        Disconnect profile events when the profile is hidden.
        """
        if pick_connection is not None:
            self.fig.canvas.mpl_disconnect(pick_connection)
        if hover_connection is not None:
            self.fig.canvas.mpl_disconnect(hover_connection)

    def update_log_axes_buttons(self, *args, **kwargs):
        """
        Update the state (value and color) of toolbar log axes buttons when
        axes or dimensions are swapped.
        """
        if self.toolbar is not None:
            self.toolbar.update_log_axes_buttons(*args, **kwargs)

    def update_norm_button(self, *args, **kwargs):
        """
        Change state of norm button in toolbar.
        """
        if self.toolbar is not None:
            self.toolbar.update_norm_button(*args, **kwargs)

    def home_view(self, *args, **kwargs):
        self.toolbar.home_view()

    def pan_view(self, *args, **kwargs):
        self.toolbar.pan_view()

    def zoom_view(self, *args, **kwargs):
        self.toolbar.zoom_view()

    def save_view(self, *args, **kwargs):
        self.toolbar.save_view()
