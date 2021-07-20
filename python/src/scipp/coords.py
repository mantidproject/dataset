# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock, Jan-Lukas Wynen

import inspect
from typing import Union, List, Dict, Tuple, Callable
from . import Variable, DataArray, Dataset, bins, VariableError


def _argnames(func):
    spec = inspect.getfullargspec(func)
    if spec.varargs is not None or spec.varkw is not None:
        raise ValueError(
            "Function with variable arguments not allowed in conversion graph."
        )
    return spec.args + spec.kwonlyargs


class Graph:
    def __init__(self, graph):
        # Keys in graph may be tuple to define multiple outputs
        self._graph = {}
        for key in graph:
            for k in [key] if isinstance(key, str) else key:
                self._graph[k] = graph[key]

    def __getitem__(self, name):
        return self._graph[name]

    def __contains__(self, name):
        return name in self._graph

    def show(self):
        from graphviz import Digraph
        dot = Digraph(strict=True)
        dot.attr('node', shape='box')
        for output, producer in self._graph.items():
            if isinstance(producer, str):  # rename
                dot.edge(producer, output)
            else:
                name = f'{producer.__name__}(...)'
                dot.node(name,
                         shape='ellipse',
                         style='filled',
                         color='lightgrey')
                dot.edge(name, output)
                argnames = _argnames(producer)
                for arg in argnames:
                    dot.edge(arg, name)
        return dot


def _consume_coord(obj, name):
    if name in obj.coords:
        obj.attrs[name] = obj.coords[name]
        del obj.coords[name]
    return obj.attrs[name]


def _produce_coord(obj, name):
    if name in obj.attrs:
        obj.coords[name] = obj.attrs[name]
        del obj.attrs[name]
    return obj.coords[name]


class CoordTransform:
    Graph = Dict[Union[str, Tuple[str, ...]], Union[str, Callable]]

    def __init__(self, obj, *, graph, outputs):
        self.obj = obj.copy(deep=False)
        # TODO We manually shallow-copy the buffer, until we have a better
        # solution for how shallow copies also shallow-copy event buffers.
        if self.obj.bins is not None:
            self.obj.data = bins(**self.obj.bins.constituents)
        self._original = obj
        self._events_copied = False
        self._rename = {}
        self._memo = []  # names of product for cycle detection
        self._aliases = []  # names that alias other names
        self._consumed = []  # names that have been consumed
        self._outputs = outputs  # names of outputs
        self._graph = graph

    def _add_event_coord(self, key, coord):
        try:
            self.obj.bins.coords[key] = coord
        except VariableError:  # Thrown on mismatching bin indices, e.g. slice
            self.obj.data = self.obj.data.copy()
            self.obj.bins.coords[key] = coord

    def _add_event_coords(self, coords):
        for key, coord in coords.items():
            # Non-binned coord should be duplicate of dense handling above,
            # if present => ignored.
            if coord.bins is not None:
                self._add_event_coord(key, coord)

    def _add_coord(self, *, name):
        if name in self.obj.meta:
            return _produce_coord(self.obj, name)
        if isinstance(self._graph[name], str):
            self._aliases.append(name)
            out = self._get_coord(self._graph[name])
            if self.obj.bins is not None:
                # Calls to _get_coord for dense coord handling take care of
                # recursion and add also event coords, here and below we thus
                # simply consume the event coord.
                if self._graph[name] in self.obj.meta:
                    out_bins = _consume_coord(self.obj.bins, self._graph[name])
            dim = (self._graph[name], )
        else:
            func = self._graph[name]
            argnames = _argnames(func)
            args = {arg: self._get_coord(arg) for arg in argnames}
            out = func(**args)
            if self.obj.bins is not None:
                args.update({
                    arg: _consume_coord(self.obj.bins, arg)
                    for arg in argnames if arg in self.obj.bins.meta
                })
                out_bins = func(**args)
            dim = tuple(argnames)
        if isinstance(out, Variable):
            out = {name: out}
        self._rename.setdefault(dim, []).extend(out.keys())
        for key, coord in out.items():
            self.obj.coords[key] = coord
        if self.obj.bins is not None:
            if isinstance(out_bins, Variable):
                out_bins = {name: out_bins}
            self._add_event_coords(out_bins)

    def _get_coord(self, name):
        if name in self.obj.meta:
            self._consumed.append(name)
            if name in self._outputs:
                return self.obj.meta[name]
            else:
                return _consume_coord(self.obj, name)
        else:
            if name in self._memo:
                raise ValueError("Cycle detected in conversion graph.")
            self._memo.append(name)
            self._add_coord(name=name)
            return self._get_coord(name)

    def _del_attr(self, name):
        if name in self.obj.attrs:
            del self.obj.attrs[name]
        if self.obj.bins is not None:
            if name in self.obj.bins.attrs:
                del self.obj.bins.attrs[name]

    def _del_coord(self, name):
        if name in self.obj.coords:
            del self.obj.coords[name]
        if self.obj.bins is not None:
            if name in self.obj.bins.coords:
                del self.obj.bins.coords[name]

    def finalize(self, *, include_aliases, rename_dims, keep_intermediate,
                 keep_inputs):
        for name in self._outputs:
            self._add_coord(name=name)
        if not include_aliases:
            for name in self._aliases:
                self._del_attr(name)
        for name in self._consumed:
            if name not in self.obj.attrs:
                continue
            if name in self._original.meta:
                if not keep_inputs:
                    self._del_attr(name)
            elif not keep_intermediate:
                self._del_attr(name)
        unconsumed = set(self.obj.coords) - set(self._original.meta) - set(
            self._outputs)
        for name in unconsumed:
            if name not in self._graph:
                self._del_coord(name)
        if rename_dims:
            blacklist = _get_splitting_nodes(self._rename)
            for key, val in self._rename.items():
                found = [k for k in key if k in self.obj.dims]
                # rename if exactly one input is dimension-coord
                if len(val) == 1 and len(
                        found) == 1 and found[0] not in blacklist:
                    self.obj = self.obj.rename_dims({found[0]: val[0]})
        return self.obj


def _get_splitting_nodes(graph):
    nodes = {}
    for key in graph:
        for start in key:
            nodes[start] = nodes.get(start, 0) + 1
    return [node for node in nodes if nodes[node] > 1]


def _transform_data_array(obj: DataArray, coords, graph: dict, *,
                          kwargs) -> DataArray:
    transform = CoordTransform(
        obj,
        graph=Graph(graph),
        outputs=[coords] if isinstance(coords, str) else coords)
    return transform.finalize(**kwargs)


def _transform_dataset(obj: Dataset, coords, graph: dict, *,
                       kwargs) -> Dataset:
    # Note the inefficiency here in datasets with multiple items: Coord
    # transform is repeated for every item rather than sharing what is
    # possible. Implementing this would be tricky and likely error-prone,
    # since different items may have different attributes. Unless we have
    # clear performance requirements we therefore go with the safe and
    # simple solution
    return Dataset(
        data={
            name: _transform_data_array(
                obj[name], coords=coords, graph=graph, kwargs=kwargs)
            for name in obj
        })


def transform_coords(x: Union[DataArray, Dataset],
                     coords: Union[str, List[str]],
                     graph: CoordTransform.Graph,
                     *,
                     rename_dims=True,
                     include_aliases=True,
                     keep_intermediate=True,
                     keep_inputs=True) -> Union[DataArray, Dataset]:
    """Compute new coords based on transformation of input coords.

    :param x: Input object with coords.
    :param coords: Name or list of names of desired output coords.
    :param graph: A graph defining how new coords can be computed from existing
                  coords. This may be done in multiple steps.
    :param rename_dims: Rename dimensions if products of dimension coord are
                        fully consumed and consumer consumes exectly one
                        dimension coordinate. Default is True.
    :param include_aliases: If True, aliases for coords defined in graph are
                            included in the output. Default is False.
    :param keep_intermediate: Keep attributes created as intermediate results.
                              Default is True.
    :param keep_inputs: Keep consumed input coordinates or attributes.
                        Default is True.
    :return: New object with desired coords. Existing data and meta-data is
             shallow-copied.
    """
    kwargs = {
        'rename_dims': rename_dims,
        'include_aliases': include_aliases,
        'keep_intermediate': keep_intermediate,
        'keep_inputs': keep_inputs
    }
    if isinstance(x, DataArray):
        return _transform_data_array(x,
                                     coords=coords,
                                     graph=graph,
                                     kwargs=kwargs)
    else:
        return _transform_dataset(x, coords=coords, graph=graph, kwargs=kwargs)


def show_graph(graph):
    return Graph(graph).show()
