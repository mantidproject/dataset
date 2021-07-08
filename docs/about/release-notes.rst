.. _release-notes:

Release Notes
=============

Since v0.7
----------

Features
~~~~~~~~

* Added ``sizes`` argument to ``zeros``, ``ones``, and ``empty`` variable creation functions `#1951 <https://github.com/scipp/scipp/pull/1951>`_.
* Slicing syntax now supports ellipsis, e.g., ``da.data[...] = var`` `#1960 <https://github.com/scipp/scipp/pull/1960>`_.
* Added bound method equivalents to many free functions which take a single Variable or DataArray `#1969 <https://github.com/scipp/scipp/pull/1969>`_.
* Variables can now be constructed directly from multi dimensional lists and tuples `#1977 <https://github.com/scipp/scipp/pull/1977>`_.

Breaking changes
~~~~~~~~~~~~~~~~

.. include:: <isonum.txt>

* Keyword arg ``to`` of ``flatten`` function renamed to ``dim``, far consistency with ``fold``.
* Changed names of arithmetic functions to match numpy's names: ``plus`` |rarr| :func:`scipp.add`, ``minus`` |rarr| :func:`scipp.subtract`, ``times`` |rarr| :func:`scipp.multiply` `#1999 <https://github.com/scipp/scipp/pull/1999>`_.
* Changed Variable init method, all arguments are keyword-only, special overloads which default-initialize data were removed `#1994 <https://github.com/scipp/scipp/pull/1994>`_.
* Made arguments of many functions keyword-only, e.g. constructors of ``DataArray`` and ``Dataset``, ``bin``, and ``atan2`` `#1983 <https://github.com/scipp/scipp/pull/1983>`_, `#2008 <https://github.com/scipp/scipp/pull/2008>`_, `#2012 <https://github.com/scipp/scipp/pull/2012>`_.
* ``astype`` and ``to_unit`` now copy the input by default even when no transformation is required, use the new ``copy`` argument to avoid `#2016 <https://github.com/scipp/scipp/pull/2016>`_.

Bugfixes
~~~~~~~~

Contributors
~~~~~~~~~~~~

Owen Arnold :sup:`b, c`\ ,
Simon Heybrock :sup:`a`\ ,
Greg Tucker :sup:`a`\ ,
Neil Vaytet :sup:`a`\ ,
and Jan-Lukas :sup:`a`\ 

v0.7.0 (June 2021)
------------------

Features
~~~~~~~~

* Licence changed from GPLv3 to BSD-3-Clause.
* Can now control the position and visibility of the legend in 1d plots with ``plot(da, legend={"show": True, "loc": 4})`` `#1790 <https://github.com/scipp/scipp/pull/1790>`_.
* Added ``zeros_like``, ``ones_like`` and ``empty_like`` functions `#1864 <https://github.com/scipp/scipp/pull/1864>`_.
* ``sort`` is now considerably faster for data with more rows `#1872 <https://github.com/scipp/scipp/pull/1872>`_.
* Added numpy-like ``linspace``, ``logspace``, ``geomspace``, and ``arange`` variable creation functions `#1871 <https://github.com/scipp/scipp/pull/1871>`_.
* ``to_unit`` now supports binned data `#1841 <https://github.com/scipp/scipp/pull/1841>`_.
* ``broadcast`` now returns a view instead of copying `#1861 <https://github.com/scipp/scipp/pull/1861>`_.
* ``fold`` now returns a view instead of copying `#1861 <https://github.com/scipp/scipp/pull/1861>`_.
* ``fold`` now works for binned data `#1861 <https://github.com/scipp/scipp/pull/1861>`_.
* ``flatten`` now returns a view of the input buffer if the input data is contiguous `#1861 <https://github.com/scipp/scipp/pull/1861>`_.
* Add ``redraw`` method to plot which enables an easy update on a figure when the underlying data has been modified `#1907 <https://github.com/scipp/scipp/pull/1907>`_.
* Several improvements for work with (3-D position) vectors and (3-D rotation) matrices `#1905 <https://github.com/scipp/scipp/pull/1905>`_:

  * Add creation functions ``vector``, ``vectors``, ``matrix``, ``matrices``.
  * Direct creation and initialization of 2-D (or higher) arrays of matrices and vectors is now possible from numpy arrays.
  * Fix extremely slow initialization of array of vectors or matrices from numpy arrays.
  * The ``values`` property now returns a numpy array with ``ndim+1`` (vectors) or ``ndim+2``` (matrices) axes, with the inner 1 (vectors) or 2 (matrices) axes corresponding o the vector or matrix axes.
  * Vector or matrix element can now be accessed and modified directly using the new ``fields`` property of ``Variable``.
    The ``fields`` property provides properties ``x``, ``y``, ``z`` (for variables containing vectors) or ``xx``, ``xy``, ..., ``zz`` (for matrices) that can be read as well as set.

* Reduction operations such as ``sum`` and ``mean`` are now also multi-threaded and thus considerably faster `#1923 <https://github.com/scipp/scipp/pull/1923>`_.

Bugfixes
~~~~~~~~

* Profile plots are no longer disabled for binned data with only a single bin in the 3rd dimension `#1936 <https://github.com/scipp/scipp/pull/1936>`_.

Breaking changes
~~~~~~~~~~~~~~~~

* Scipp's data structures now behave mostly like normal nested Python objects, i.e., copies are shallow by default `#1823 <https://github.com/scipp/scipp/pull/1823>`_.
* ``filter`` and ``split`` removed. Identical functionality can be achieved using ``groupby`` and/or slicing.
* ``reshape`` has been removed. Use ``fold`` and ``flatten`` instead `#1861 <https://github.com/scipp/scipp/pull/1861>`_.
* ``geometry.x``, ``geometry.y``, and ``geometry.z`` have been removed. Use the ``Variable.fields`` property instead `#1925 <https://github.com/scipp/scipp/pull/1925>`_.

Contributors
~~~~~~~~~~~~

Owen Arnold :sup:`b, c`\, 
Simon Heybrock :sup:`a`\,
Matthew D. Jones :sup:`b, c`\,
Neil Vaytet :sup:`a`\,
and Jan-Lukas Wynen :sup:`a`\

v0.6.1 (April 2021)
-------------------

Bugfixes
~~~~~~~~

* ``map`` and ``scale`` operations as well as ``histogram`` for binned data now also work with ``datetime64`` `#1834 <https://github.com/scipp/scipp/pull/1834>`_.
* ``bin`` now works on previously binned data with 2-D edges, even if the outer dimensions(s) are not rebinned `#1836 <https://github.com/scipp/scipp/pull/1836>`_.
* ``bin`` and ``histogram`` now work with ``int32``-valued coordinates and support binning with ``int64``- or ``int32``-valued bin edges.

v0.6.0 (March 2021)
-------------------

Features
~~~~~~~~

* Add ``sizes`` properties, an ordered ``dict`` to lookup the size of a specific dimension `#1636 <https://github.com/scipp/scipp/pull/1636>`_.
* Add access to data, coords, attrs, and masks of bins via the ``bins`` property, e.g., ``obj.bins.coords['x']`` `#1752 <https://github.com/scipp/scipp/pull/1752>`_.
* Add more functions for data arrays that were previously available only for variables `#1638 <https://github.com/scipp/scipp/pull/1638>`_ `#1660 <https://github.com/scipp/scipp/pull/1660>`_.
* Add named versions of operators such as ``logical_and`` `#1660 <https://github.com/scipp/scipp/pull/1660>`_.
* Add ``to_unit`` to converter variables, e.g., between ``m`` and ``mm`` `#1756 <https://github.com/scipp/scipp/pull/1756>`_.
* Add modulo operations `#1660 <https://github.com/scipp/scipp/pull/1660>`_.
* Add scaling operations for Variables of type ``vector_3_float64``.
* ``sum`` and ``mean`` implemented for Variables of type ``vector_3_float64``.
* Add ``fold`` and ``flatten`` which allow to reshape dimensions of a Variable or DataArray `#1676 <https://github.com/scipp/scipp/pull/1676>`_.
* It is now possible to reshape a Variable also with ``sc.reshape(var, sizes={'x': 2, 'y': 3})``, in addition to ``sc.reshape(var, dims=['x', 'y'], shape=(2, 3))``.
* Add ``ones`` and ``empty`` creation functions, similar to what is known from numpy `#1732 <https://github.com/scipp/scipp/pull/1732>`_.
* ``scipp.neutron`` has been removed and is replaced by `scippneutron <https://scipp.github.io/scippneutron>`_
* ``scipp.neutron`` (now ``scippneutron``)

  * Support unit conversion to energy transfer, for inelastic TOF experiments `#1635 <https://github.com/scipp/scipp/pull/1635>`_.
  * Support loading/converting Mantid ``WorkspaceGroup``, this will produce a ``dict`` of data arrays `#1654 <https://github.com/scipp/scipp/pull/1654>`_.
  * Fixes to support loading/converting ``McStasNexus`` files `#1659 <https://github.com/scipp/scipp/pull/1659>`_.
* ``isclose`` added (``is_approx`` removed). Fuzzy data comparison ``isclose`` is an analogue to numpy's ``isclose``
* ``stddevs`` added `#1762 <https://github.com/scipp/scipp/pull/1762>`_.

* Support for datetime

  * ``sc.dtype.datetime64`` with copy-less casting between numpy and scipp where possible. `#1639 <https://github.com/scipp/scipp/pull/1639>`_
  * Binning with datetime edges `#1739 <https://github.com/scipp/scipp/pull/1739>`_

Breaking changes
~~~~~~~~~~~~~~~~

* The ``plot`` module has been renamed to ``plotting``, and plotting is now achieved via ``sc.plot(data)``. Wrapper functions ``superplot``, ``image``, and ``scatter3d`` have been dropped `#1696 <https://github.com/scipp/scipp/pull/1696>`_.
* Properties ``dim``, ``begin``, ``end``, and ``data`` of the ``bins`` property of and object are now accessible as a dict via ``obj.bins.constituents``.
* ``scipp.neutron`` has been removed and is replaced by `scippneutron <https://scipp.github.io/scippneutron>`_
* ``scipp.neutron`` (now ``scippneutron``)
* ``is_equal`` renamed ``identical``
* ``is_linspace`` renamed ``islinspace``
* ``is_sorted`` renamed ``issorted``
* ``is_approx`` superseded by ``isclose``. ``is_approx`` removed.
* Removed support for facility-specific ``convert_with_calibration`` and ``load_calibration``

Contributors
~~~~~~~~~~~~

Matthew Andrew,
Owen Arnold,
Simon Heybrock,
Matthew D. Jones,
Andrew McCluskey,
Neil Vaytet,
and Jan-Lukas Wynen

v0.5.0 (January 2021)
---------------------

Features
~~~~~~~~

* New ``profile`` plotting functionality where one of the slider dimensions can be displayed as a profile in a subplot (1-D and 2-D projections only).
* Sliders have a thickness slider associated with them and can be used to show slices of arbitrary thickness.
* Can hide/show individual masks on plots.
* Can toggle log scale of axes and colorbar with buttons in figure toolbar.
* Add binned data support, replacing "event list" dtypes as well as "realign" support.
* Plotting of event data (binned data) with dynamic resampling with "infinite zoom" functionality.
* Value-based slicing support.
* Possibility to plot Scipp objects using ``my_data_array.plot()`` in addition to the classical ``plot()`` free function.
* Support for saving and loading scipp data structures to HDF5.
* More functions such as ``nanmean`` for better handling of special values such as ``INF`` and ``NaN``.
* TBB (multi-threading) support for MacOS.
* ``scipp.neutron``

  * Improved instrument view, e.g., with buttons to align camera with an axis.
  * Experiment logs (previously using Mantid's ``Run``) are now represented as native scipp objects, e.g., as scalar attributes holding a data array representing a time-series such as a temperature log.
  * Support conversion of ``mantid.MaskWorkspace``.

Breaking changes
~~~~~~~~~~~~~~~~

* ``Dataset`` does not have a ``masks`` property any more.
  Use ``ds['item'].masks`` instead.
* ``Dataset`` does not support attributes any more.
* ``DataArray`` and dataset item attributes are now are now handled as "unaligned" coords.
  Use ``ds['item'].coords`` or ``array.attrs`` to access these.
* API for log scale on axes and colors has changed.
  Use ``plot(da, scale={'tof': 'log'})`` to set a log scale on a coordinate axis, and use ``plot(da, norm='log')`` to have a log image colorscale or a log y axis on a 1d plot.
* ``vmin`` and ``vmax`` now represent absolute values instead of exponents when ``norm='log'``.
* The ``ipympl`` matplotlib backend is now required for using inside Jupyter notebooks.
  This has been added as a dependency.
  It is also the only interactive backend that works in JupyterLab.
* Removed support for ``event_list`` ``dtype``, use binned data instead.
* Removed support for "realigned" data. This is replaced by the more flexible and generic support for "binned" data.

Contributors
~~~~~~~~~~~~

Matthew Andrew,
Owen Arnold,
Thibault Chatel,
Simon Heybrock,
Matthew D. Jones,
Daniel Nixon,
Piotr Rozyczko,
Neil Vaytet,
and Jan-Lukas Wynen

v0.4 (July 2020)
----------------

Features
~~~~~~~~

* New realign functionality.
* Support for event-filtering.
* Support for subtraction and addition for (realigned) event data.
* Non-range slicing changed to preserve coords as attrs rather than dropping
* ``scipp.neutron``: Instrument view with advanced geometry support, showing correct pixel shapes.
* Instrument view working on doc pages.
* Made it simpler to add new ``dtype`` and support ``transform`` for all types.
* Comparison functions such as ``less``, ``greater_equal``, ...
* ``all`` and ``any`` can work over all dimensions as well as explicitly provided dimension argument
* It is now possible to convert between Scipp objects and Python dictionaries using ``to_dict`` and ``from_dict``.
* New functions ``collapse`` and ``slices`` can be use to split one or more dimensions of a DataArray to a dict of DataArrays.
* You can now inspect the global object list of via the ``repr`` for scipp showing Datasets, DataArrays and Variables
* Internal cleanup and documentation additions.

Noteable bug fixes
~~~~~~~~~~~~~~~~~~

* Several fixes in the plotting (non-regular bins, colorbar limits, axes tick labels from unaligned coordinates, etc...)

Breaking changes
~~~~~~~~~~~~~~~~

* Coord and attributes names for neutron data have been standardized, now using hyphens instead of underscore, except for subscripts. Affected examples: ``pulse-time`` (previously ``pulse_times``), ``source-position`` (previously ``source_position``), ``sample-position`` (previously ``sample_position``), ``detector-info`` (previously ``detector_info``).
* ``scipp.neutron.load`` must use ``advanced_geometry=True`` option for loading ``detector-info`` and pixel shapes.
* Normalization of event data cannot be done directly any more, must use ``realign``.
* Plotting variances in 2D has been removed, and the API for using ``matplotlib`` axes has been simplified slightly, since we no longer have axes for variances:

  * Before: ``plot(..., mpl_axes={"ax": myax0, "cax": myax1})``
  * After: ``plot(..., ax=myax0, cax=myax1)``
* Plot with keyword argument ``collapse`` has been removed in favor of two more generic free functions that return a ``dict`` of data arrays that can then directly be passed to the ``plot`` function:

  * ``collapse(d, keep='x')`` slices all dimensions away to keep only ``'x'``, thus always returning 1D slices.
  * ``slices(d, dim='x')`` slices along dimension ``'x'``, returning slices with ``ndim-1`` dimensions contaiing all dimensions other than ``'x'``.

Contributors
~~~~~~~~~~~~

Owen Arnold,
David Fairbrother,
Simon Heybrock,
Daniel Nixon,
Pawel Ptasznik,
Piotr Rozyczko,
and Neil Vaytet


v0.3 (March 2020)
-----------------

* Many bug fixes and small additions
* Multi-threading with TBB for many operations.
* Performance improvements in hotspots
* Remove ``Dim`` labels in favor of plain strings. Connected to this, the ``labels`` property for data arrays and datasets has been removed. Use ``coords`` instead.
* Start to support ``out`` arguments (not everywhere yet)
* ``scipp.neutron``: Instrument view added

Contributors in this release:
Owen Arnold,
Simon Heybrock,
Daniel Nixon,
Dimitar Tasev,
and Neil Vaytet


v0.2 (December 2019)
--------------------

* Support for masks stored in ``DataArray`` and ``Dataset``.

* Support for ``groupby``, implementing a split-apply-combine approach as known from pandas.

* Enhanced support for event data:

  * Histogramming with "weighted" data.
  * Multiplication/division operators between event data and histogram.

* Enhanced plotting support:

  * Now focussing on ``matplotlib``.
  * Multi-dimensional plots with interactive sliders, and much more.

* Significant performance improvements for majority of operations. Typically performance is now in the same ballpark as what the memory bandwidth on a single CPU core can support.

* Fancy ``_repr_html_`` for quick views of datasets in Jupyter notebooks.

* Conda packages now also available for Windows.

* ``scipp.neutron`` gets improved converters from Mantid, supporting neutron monitors, sample information, and run information stored as attributes.

Contributors in this release:
Owen Arnold,
Igor Gudich,
Simon Heybrock,
Daniel Nixon,
Dimitar Tasev,
and Neil Vaytet


v0.1 (September 2019)
---------------------

This is the first official release of ``scipp``.
It is not yet meant for production-use, but marks a big step for us in terms of usability and features.
The API may change without notice in future releases.

Features:

* All key data structures (``Variable``, ``DataArray``, and ``Dataset``).
* Slicing.
* Basic arithmetic operations.
* Physical units.
* Propagation of uncertainties.
* Event data.

Limitations:

* Limited performance and no parallelization.
* Numerous "edge cases" not supported yet.
* While tested, probably far from bug-free.

Contributing Organizations
--------------------------
* :sup:`a`\  `European Spallation Source ERIC <https://europeanspallationsource.se/>`_, Sweden
* :sup:`b`\  `Science and Technology Facilities Council <https://www.ukri.org/councils/stfc/>`_, UK
* :sup:`c`\  `Tessella <https://www.tessella.com/>`_, UK
