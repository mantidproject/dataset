.. _faq:

Frequently Asked Questions
==========================

Specific help with using scipp
------------------------------

For help on specific issues with using **scipp**, you should first visit
`this <https://github.com/scipp/scipp/issues?utf8=%E2%9C%93&q=label%3Aquestion>`_
page to see if the problem you are facing has already been met/solved in the community.

If you cannot find an answer, you can ask a new question by
`opening <https://github.com/scipp/scipp/issues/new?assignees=&labels=question&template=question.md&title=>`_
a new |QuestionLabel|_ issue.

.. |QuestionLabel| image:: ../images/question.png
.. _QuestionLabel: https://github.com/scipp/scipp/issues/new?assignees=&labels=question&template=question.md&title=

General
-------

Why is xarray not enough?
~~~~~~~~~~~~~~~~~~~~~~~~~

For our application (handling of neutron-scattering data, which is so far mostly done using `Mantid <https://mantidproject.org>`_), xarray is currently missing a number of features:

- Handling of physical units.
- Propagation of uncertainties.
- Support for histograms, i.e., bin-edge axes, which are by 1 longer than the data extent.
- Support for event data, a particular form of sparse data.
  More concretely, this is essentially a 1-D (or N-D) array of random-length lists, with very small list entries.
  This type of data arises in time-resolved detection of neutrons in pixelated detectors.
- Written in C++ for performance opportunities, in particular also when interfacing with our extensive existing C++ codebase.

Why are you not just contributing to xarray?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is a valid criticism and at times we still feel uncertain about this choice.
The decision was made in the context of a larger project which needed to come to a conclusion within a couple of years, including all the items listed in the previous FAQ entry.
Essentially we felt that the list of additional requirements on top of what xarray provides was too long.
Effecting/contributing such fundamental changes to an existing framework is a long process and likely not obtained within the given time frame.
Furthermore, some of the requirements are unlikely to be obtainable within xarray.

We should note that at least some of our additional requirements, in particular physical units, are being pursued also by the xarray developers.

Plotting
--------

How can I set axis limits when creating a plot?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This can be achieved indirectly with scipp's `generic slicing functionality <../user-guide/slicing.rst>`_ and `label-based indexing <../user-guide/slicing.ipynb#Label-based-indexing>`_ in particular.
Example:

.. code-block:: python

  array.plot()  # plot with full x range
  array['x', 100:200].plot()  # plot 100 points starting at offset 100
  start = 1.2 * sc.Unit('m')
  stop = 1.3 * sc.Unit('m')
  array['x', start:stop].plot()  # plot everything between 1.2 and 1.3 meters

Installation
------------

I can't import scipp!
~~~~~~~~~~~~~~~~~~~~~

On Windows, after installing ``scipp`` using ``conda``, attempting to ``import scipp`` may sometimes fail with

.. code-block:: python

  > import scipp as sc
  > ImportError: DLL load failed: The specified module could not be found.

This issue is Windows specific and fixing it requires downloading and installing a recent version of the Microsoft Visual C++ Redistributable for
Visual Studio 2019.
It can be downloaded from `Microsoft's official site <https://support.microsoft.com/en-us/topic/the-latest-supported-visual-c-downloads-2647da03-1eea-4433-9aff-95f26a218cc0>`_.
