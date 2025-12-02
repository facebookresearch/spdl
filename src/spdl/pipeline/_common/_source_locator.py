# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Module for locating source code of functions, callable objects, and wrapped functions.
"""

import functools
import inspect
from typing import Any, NamedTuple


def _get_qualified_name(obj: Any) -> str:
    """Get the fully qualified name of an object."""
    if hasattr(obj, "__module__") and hasattr(obj, "__qualname__"):
        if obj.__module__ == "__main__":
            return obj.__qualname__
        return f"{obj.__module__}.{obj.__qualname__}"

    if hasattr(obj, "__module__") and hasattr(obj, "__name__"):
        if obj.__module__ == "__main__":
            return obj.__name__
        return f"{obj.__module__}.{obj.__name__}"

    if hasattr(obj, "__name__"):
        return obj.__name__

    return str(obj)


class SourceLocation(NamedTuple):
    name: str
    file_path: str | None
    line_number: int | None
    partial_args: tuple[Any, ...]
    partial_kwargs: dict[str, Any]


_BUILTIN_TYPES: tuple[type[Any], ...] = (
    tuple,
    list,
    dict,
    str,
    bool,
    int,
    float,
    set,
    frozenset,
    bytes,
    bytearray,
    memoryview,
)


def locate_source(func: Any) -> SourceLocation:
    """
    Locate the source code of a function or callable object.

    Args:
        func: A function, callable object, or functools.partial instance

    Returns:
        SourceLocation containing file path, line number, name, and any partial args
    """
    partial_args: list[Any] = []
    partial_kwargs: dict[str, Any] = {}

    # Unwrap nested functools.partial instances
    while isinstance(func, functools.partial):
        if func.args:
            partial_args = list(func.args) + partial_args
        if func.keywords:
            partial_kwargs = {**func.keywords, **partial_kwargs}
        func = func.func

    if isinstance(func, _BUILTIN_TYPES):
        return SourceLocation(
            file_path=None,
            line_number=None,
            name=type(func).__name__,
            partial_args=tuple(partial_args),
            partial_kwargs=partial_kwargs,
        )

    # Determine the actual callable and get its name
    if inspect.isfunction(func) or inspect.ismethod(func):
        # Regular function, generator, async function, async generator
        target = func
        name = _get_qualified_name(func)
    elif inspect.isbuiltin(func):
        # Built-in function (like len, print, etc.)
        target = func
        name = _get_qualified_name(func)
    elif inspect.isclass(func):
        # Class itself
        target = func
        name = _get_qualified_name(func)
    elif callable(func):
        # Callable object - get the class
        target = func.__class__
        name = _get_qualified_name(target)
    else:
        # Fallback
        name = "unknown"
        target = func

    file_path = None
    try:
        file_path = inspect.getsourcefile(target) or inspect.getfile(target)
    except (TypeError, OSError):
        pass

    line_number = None
    try:
        _, line_number = inspect.getsourcelines(target)
    except (TypeError, OSError):
        pass

    return SourceLocation(
        file_path=file_path,
        line_number=line_number,
        name=name,
        partial_args=tuple(partial_args),
        partial_kwargs=partial_kwargs,
    )
