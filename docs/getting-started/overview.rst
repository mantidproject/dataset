.. _overview:

Overview
========

The data array concept
----------------------

The core data structure of scipp is :py:class:`scipp.DataArray`.
A data array is essentially a multi-dimensional array with associated dicts of coordinates, masks, and attributes.
For a more detailed explanation we refer to the `documentation of DataArray <../user-guide/data-structures.rst#DataArray>`_.

Scipp labels dimensions and their associated coordinates with labels such as ``'x'``, ``'temperature'``, or ``'wavelength'``.

Operations between data arrays "align" data items based on their names, dimension labels, and coordinate values.
That is, if names or coordinates do not match, operations fail.
Missing dimensions in the operands are automatically `broadcast <https://numpy.org/doc/stable/user/basics.broadcasting.html>`_.
In general scipp does not support automatic re-alignment of data.

Data arrays with aligned coordinates can be combined into datasets.
A dataset is essentially a dict-like container of data arrays with common associated coordinates.


Physical units
--------------

Data in a data array as well as coordinates are associated with a physical unit.
The unit is accessed using the ``unit`` property.
All operations take the unit into account:

- Operations fail if the units of coordinates do not match.
- Operations such as addition fail if the units of the data items do not match.
- Operations such as multiplication produce an output dataset with a new unit for each data item, e.g., :math:`m^{2}` results from multiplication of two data items with unit :math:`m`.


Variances and propagation of uncertainties
------------------------------------------

Data in a data array as well as coordinates support optional variances in addition to their values.
The variances are accessed using the ``variances`` property and have the same shape and dtype as the ``values`` array.
All operations take the variances into account:

- Operations fail if the variances of coordinates are not identical, element by element.
  For the future, we are considering supporting inexact matching based on variances of coordinates but currently this is not implemented.
- Operations such as addition or multiplication propagate the errors to the output.
  An overview of the method can be found in `Wikipedia: Propagation of uncertainty <https://en.wikipedia.org/wiki/Propagation_of_uncertainty>`_.
  The implemented mechanism assumes uncorrelated data.


Scattered data (event data)
---------------------------

Scipp supports event data in form of arrays of variable-length tables.
Conceptually, we distinguish *dense* and *binned* data.

- Dense data is the common case of a regular, e.g., 2-D, grid of data points.
- Binned (event) data (as supported by scipp) is semi-regular data, i.e., a N-D array/grid of variable-length tables.
  That is, only the internal "dimension" of the tables (event lists) are irregular.

The target application of this is measuring random *events* in an array of detector pixels.
In contrast to a regular image sensor, which may produce a 2-D image at fixed time intervals (which could be handled as 3-D data), each detector pixel will see a different event rate (photons or neutrons) and is "read out" at uncorrelated times.

Scipp handles such event data by supporting *binned data*, which we explicitly distinguish from *histogrammed data*.
Binned data can be converted into histogrammed data by summing over all the events in a bin.


Histograms and bin-edge coordinates
-----------------------------------

Coordinates for one or more of the dimensions in a data array can represent bin edges.
The extent of the coordinate in that dimension exceeds the data extent (in that dimension) by 1.
Bin-edge coordinates frequently arise when we histogram event-based data.
The event data described above can be used to produce such a histogram.

Bin-edge coordinates are handled naturally by operations with datasets.
Most operations on individual data elements are unaffected.
Other operations such as ``concatenate`` take the edges into account and ensure that the resulting concatenated edge coordinate is well-defined.
There are also a number of operations specific to data with bin-edges, such as ``rebin``.
