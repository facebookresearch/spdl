# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Implement :py:func:`fuse`, which executes SPDL IO functions in one native call.

While the fused function runs, sources (paths and byte buffers) are
symbolic values, and SPDL IO calls and attribute accesses on their results
are recorded as operations of a native program (:py:class:`_Graph`).
The program is executed when Python needs its results (:py:class:`_LazyGraph`),
which is usually only when the function returns. When a call ran as one
native call, the next call with the same argument structure traces the
function, and the traced program (:py:class:`_Program`) is reused afterwards
without executing the function.
"""

__all__ = [
    "fuse",
    "_active_graph",
    "_actual",
    "_fallback",
    "_is_symbol",
    "_record_open_source",
    "_suspend",
    "_unwrap_frames",
    "_unwrap_materialized",
    "_is_traced",
    "_packets_info",
    "_record_open",
    "_record_bsf",
    "_record_decode",
    "_record_sample_decode",
    "_record_convert",
    "_record_cuda",
    "_record_post",
]

import contextlib
import contextvars
import dataclasses
import functools
import logging
import operator
import os
import sys
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import (
    Any,
    NoReturn,
    overload,
    ParamSpec,
    TYPE_CHECKING,
    TypeGuard,
    TypeVar,
)

from ._common import _resolve_src
from .lib import _libspdl

if TYPE_CHECKING:
    _Frames = _libspdl.AudioFrames | _libspdl.VideoFrames | _libspdl.ImageFrames

try:
    from .fb import _is_fusion_disabled  # type: ignore[import-not-found]
except ImportError:

    def _is_fusion_disabled(name: str) -> bool:
        """Whether fusion is disabled for the function ``name``.

        Builds with internal extensions provide a switch to turn off fusion
        without changing code. It is read once, when the function is
        decorated. Without the extensions, fusion is always enabled.
        """
        return False


_LG: logging.Logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")

_SOURCE_KEY = object()

# How an attribute of a traced value is exposed: as a property or a method,
# and what kind of value it produces.
_PROP = "prop"
_METHOD = "method"
_AttrSpec = tuple[str, str, str | None]


def _meta(*names: str) -> dict[str, _AttrSpec]:
    return {n: (_PROP, "meta", None) for n in names}


def _meta_methods(*names: str) -> dict[str, _AttrSpec]:
    return {n: (_METHOD, "meta", None) for n in names}


def _codec(attr: str, media: str) -> dict[str, _AttrSpec]:
    return {attr: (_PROP, "codec", media)}


def _get_attr_specs() -> dict[tuple[str, str | None], dict[str, _AttrSpec]]:
    return {
        ("demuxer", None): {
            **_meta_methods("has_audio"),
            **_meta("audio_stream_index", "video_stream_index"),
            **_codec("audio_codec", "audio"),
            **_codec("video_codec", "video"),
            **_codec("image_codec", "image"),
        },
        ("packets", "audio"): {
            **_meta("timestamp", "sample_rate", "num_channels"),
            **_codec("codec", "audio"),
        },
        ("packets", "video"): {
            **_meta("timestamp", "pix_fmt", "width", "height", "frame_rate"),
            **_meta_methods("get_timestamps"),
            **_codec("codec", "video"),
        },
        ("packets", "image"): {
            **_meta("pix_fmt", "width", "height"),
            **_codec("codec", "image"),
        },
        ("frames", "audio"): _meta(
            "num_frames", "sample_rate", "num_channels", "sample_fmt"
        ),
        ("frames", "video"): {
            **_meta(
                "num_frames", "num_planes", "width", "height", "pix_fmt", "time_base"
            ),
            **_meta_methods("get_timestamps", "get_pts"),
        },
        ("frames", "image"): _meta("num_planes", "width", "height", "pix_fmt", "pts"),
        ("codec", "audio"): _meta(
            "name",
            "sample_rate",
            "num_channels",
            "sample_fmt",
            "time_base",
            "channel_layout",
        ),
        ("codec", "video"): _meta(
            "name",
            "width",
            "height",
            "pix_fmt",
            "frame_rate",
            "time_base",
            "sample_aspect_ratio",
        ),
        ("codec", "image"): _meta(
            "name", "width", "height", "pix_fmt", "time_base", "sample_aspect_ratio"
        ),
    }


_OUTPUT_KINDS = ("buffer", "cuda_buffer", "packets", "frames", "codec", "meta")

# Attributes through which other libraries read the content of a buffer.
_ARRAY_PROTOCOLS = (
    "__array_interface__",
    "__cuda_array_interface__",
    "__array__",
    "__dlpack__",
    "__dlpack_device__",
)

_CONSTANT_TYPES = (type(None), bool, int, float, str, bytes)

# Kinds of values that are moved out of the native registers when Python
# needs them. (Metadata and codecs are copied.)
_MOVED_KINDS = ("buffer", "cuda_buffer", "packets", "frames")


class _GraphBreak(BaseException):
    """Raised during tracing when Python code needs a value of the execution.

    Derived from ``BaseException`` so that ``except Exception`` in user code
    does not swallow it.
    """


class _UsageError(Exception):
    """An error in the use of SPDL IO that is raised without ``fuse`` as well.

    Raised as is while recording, rather than as a failure of fusion.
    """


class _UsageTypeError(_UsageError, TypeError):
    pass


class _UsageValueError(_UsageError, ValueError):
    pass


class _FusionError(BaseException):
    """Raised when recording a call fails, as opposed to executing it.

    Derived from ``BaseException`` so that ``except Exception`` in user code
    does not swallow it. The fused function then runs without recording.
    """

    def __init__(self, error: Exception) -> None:
        super().__init__(f"{type(error).__name__}: {error}")
        self.error = error


def _recording(fn: Callable[P, T]) -> Callable[P, T]:
    """Raise errors of recording a call while a fused function runs as ``_FusionError``.

    Errors of executing the recorded operations (e.g. when recording needs a
    value of the execution) and errors while tracing are raised as they are.
    """

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return fn(*args, **kwargs)
        except _UsageError:
            raise
        except Exception as e:
            g = _ACTIVE.get()
            if isinstance(g, _LazyGraph) and not g.failed:
                raise _FusionError(e) from e
            raise

    return wrapper


def _unwrap(obj: object) -> object:
    return obj._graph.concrete(obj) if isinstance(obj, _Symbol) else obj


class _Symbol:
    """A symbolic value produced while tracing a fused function."""

    __slots__ = (
        "_graph",
        "_reg",
        "_kind",
        "_media",
        "_timestamp",
        "_consumed",
        "__weakref__",
    )

    def __init__(
        self,
        graph: "_Graph",
        reg: int,
        kind: str,
        media: str | None = None,
        timestamp: tuple[float, float] | None = None,
    ) -> None:
        self._graph = graph
        self._reg = reg
        self._kind = kind
        self._media = media
        self._timestamp = timestamp
        self._consumed: str | None = None

    def __repr__(self) -> str:
        media = f"{self._media}, " if self._media else ""
        return f"Symbol<{self._kind}, {media}%{self._reg}>"

    def __getattr__(self, name: str) -> Any:
        if self._kind in ("buffer", "cuda_buffer") and name in _ARRAY_PROTOCOLS:
            return getattr(self._graph.concrete(self), name)
        if name.startswith("__"):
            raise AttributeError(name)
        return self._graph.attr(self, name)

    def __getitem__(self, key: object) -> Any:
        return self._graph.slice(self, key)

    def __len__(self) -> int:
        return self._graph.length(self)

    def __bool__(self) -> bool:
        if self._kind in ("packets", "frames") and self._media != "image":
            return len(self) > 0
        return True

    def __iter__(self) -> NoReturn:
        raise TypeError(f"{self!r} is not iterable.")

    __hash__ = object.__hash__


class _MetaSymbol(_Symbol):
    """Metadata, which is only known when the fused function runs.

    Using it in Python makes the fused function execute up to this point.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph.concrete(self), name)

    def __getitem__(self, key: object) -> Any:
        return self._graph.concrete(self)[key]

    def __len__(self) -> int:
        return len(self._graph.concrete(self))

    def __bool__(self) -> bool:
        return bool(self._graph.concrete(self))

    def __iter__(self) -> Any:
        return iter(self._graph.concrete(self))

    def __contains__(self, item: object) -> bool:
        return _unwrap(item) in self._graph.concrete(self)

    def __index__(self) -> int:
        return operator.index(self._graph.concrete(self))

    def __int__(self) -> int:
        return int(self._graph.concrete(self))

    def __float__(self) -> float:
        return float(self._graph.concrete(self))

    def __str__(self) -> str:
        return str(self._graph.concrete(self))

    def __format__(self, spec: str) -> str:
        return format(self._graph.concrete(self), spec)

    def __round__(self, ndigits: int | None = None) -> Any:
        return round(self._graph.concrete(self), ndigits)

    def __neg__(self) -> Any:
        return -self._graph.concrete(self)

    def __abs__(self) -> Any:
        return abs(self._graph.concrete(self))

    def __hash__(self) -> int:
        return hash(self._graph.concrete(self))


def _forward(fn: Callable[[Any, Any], Any]) -> Callable[[_Symbol, object], Any]:
    def forward(self: _Symbol, other: object) -> Any:
        return fn(self._graph.concrete(self), _unwrap(other))

    return forward


def _reflect(fn: Callable[[Any, Any], Any]) -> Callable[[_Symbol, object], Any]:
    def reflect(self: _Symbol, other: object) -> Any:
        return fn(_unwrap(other), self._graph.concrete(self))

    return reflect


def _install_binary_operators() -> None:
    for name in ("add", "sub", "mul", "truediv", "floordiv", "mod", "pow"):
        fn = getattr(operator, name)
        setattr(_MetaSymbol, f"__{name}__", _forward(fn))
        setattr(_MetaSymbol, f"__r{name}__", _reflect(fn))
    for name in ("eq", "ne", "lt", "le", "gt", "ge"):
        setattr(_MetaSymbol, f"__{name}__", _forward(getattr(operator, name)))


_install_binary_operators()


def _source_used(self: "_SourceSymbol", *_: object, **__: object) -> NoReturn:
    raise _GraphBreak(f"{self!r} is used other than as a source of SPDL IO.")


class _SourceSymbol(_Symbol):
    """A source argument of a function traced with ``cache=True``.

    An argument is classified as a source by its type (``str``, ``bytes``
    and other buffer objects), which can be wrong. If Python code uses it in
    any way other than passing it to SPDL IO functions, the function is not
    cached, because the value would be baked into the cached program.
    """

    __slots__ = ()

    __getattr__ = _source_used
    __getitem__ = _source_used
    __len__ = _source_used
    __iter__ = _source_used
    __bool__ = _source_used
    __str__ = _source_used
    __format__ = _source_used
    __fspath__ = _source_used
    __buffer__ = _source_used
    __index__ = _source_used
    __int__ = _source_used
    __float__ = _source_used
    __eq__ = _source_used
    __ne__ = _source_used
    __add__ = _source_used
    __radd__ = _source_used
    __truediv__ = _source_used
    __rtruediv__ = _source_used
    __mod__ = _source_used
    __contains__ = _source_used

    __hash__ = object.__hash__


class _DemuxerSymbol(_Symbol):
    """Implements the interface of the native demuxer used by :py:class:`Demuxer`."""

    __slots__ = ("_source", "_eager")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The source, demux config and name, when the source is concrete.
        self._source: tuple[Any, Any, str | None] | None = None
        # The native demuxer opened for operations that cannot be recorded.
        self._eager: Any = None

    @_recording
    def demux_audio(
        self,
        window: tuple[tuple[int, int], tuple[int, int]] | None = None,
        bsf: str | None = None,
    ) -> Any:
        return self._graph.demux(self, "audio", window, bsf)

    @_recording
    def demux_video(
        self,
        window: tuple[tuple[int, int], tuple[int, int]] | None = None,
        bsf: str | None = None,
    ) -> Any:
        return self._graph.demux(self, "video", window, bsf)

    @_recording
    def demux_image(self, bsf: str | None = None) -> Any:
        return self._graph.demux(self, "image", None, bsf)

    def streaming_demux(
        self, indices: set[int], *, num_packets: int, duration: float
    ) -> Any:
        return self._graph.streaming_demux(self, indices, num_packets, duration)

    def _drop(self) -> None:
        self._graph.use(self, "demuxer", consume="closing the demuxer")


class _PostSymbol(_Symbol):
    """A buffer to be converted to an array after execution."""

    __slots__ = ("_base", "_fn")

    def __init__(self, base: _Symbol, fn: Callable[[Any], Any]) -> None:
        super().__init__(base._graph, base._reg, "post")
        self._base = base
        self._fn = fn

    def __getattr__(self, name: str) -> NoReturn:
        raise _GraphBreak(f"`{name}` of the result of `{self._fn.__name__}`")


class _Graph:
    def __init__(self) -> None:
        self.num_registers = 0
        self.inputs: list[_Symbol] = []
        # (operation, arguments, used registers, defined registers)
        self.ops: list[tuple[str, dict[str, Any], list[int], list[int]]] = []
        self._attr_specs: dict[tuple[str, str | None], dict[str, _AttrSpec]] = (
            _get_attr_specs()
        )
        # Register -> symbol, to find the values that are no longer used.
        self.symbols: dict[int, weakref.ref[_Symbol]] = {}
        # Registers of the sources passed to SPDL IO functions.
        self.opened: set[int] = set()
        # Whether executing recorded operations failed.
        self.failed = False

    def _new(
        self,
        kind: str,
        media: str | None = None,
        timestamp: tuple[float, float] | None = None,
        cls: type[_Symbol] | None = None,
    ) -> _Symbol:
        if cls is None:
            cls = _MetaSymbol if kind == "meta" else _Symbol
        sym = cls(self, self.num_registers, kind, media, timestamp)
        self.num_registers += 1
        self.symbols[sym._reg] = weakref.ref(sym)
        return sym

    def concrete(self, sym: _Symbol) -> Any:
        raise _GraphBreak(repr(sym))

    def is_materialized(self, sym: _Symbol) -> bool:
        return False

    def open_source(
        self, src: object, demux_config: object, name: str | None
    ) -> _Symbol:
        raise _GraphBreak("a source that is not an argument")

    def streaming_demux(
        self, demuxer: _Symbol, indices: set[int], num_packets: int, duration: float
    ) -> Any:
        raise _GraphBreak("streaming demuxing")

    def length(self, sym: _Symbol) -> int:
        if self.is_materialized(sym):
            return len(self.concrete(sym))
        self.use(sym, "packets", "frames")
        if sym._media == "image":
            raise TypeError(f"{sym!r} has no len().")
        return self.concrete(self.query(sym, "__len__", "meta", None))

    def _emit(
        self, operation: str, uses: list[int], defs: list[int], **kwargs: Any
    ) -> None:
        self.ops.append((operation, kwargs, uses, defs))

    def source(self) -> _Symbol:
        sym = self._new("source", cls=_SourceSymbol)
        self.inputs.append(sym)
        return sym

    def use(
        self, sym: object, *kinds: str, media: str | None = None, consume: str = ""
    ) -> _Symbol:
        if not isinstance(sym, _Symbol):
            raise TypeError(f"Expected a traced value, but found {type(sym)}.")
        if sym._graph is not self:
            raise ValueError(f"{sym!r} belongs to a different fused function.")
        if sym._kind not in kinds:
            raise _UsageTypeError(f"Expected {' or '.join(kinds)}, but found {sym!r}.")
        if media is not None and sym._media != media:
            raise _UsageTypeError(f"Expected {media} {sym._kind}, but found {sym!r}.")
        if sym._consumed:
            raise _UsageValueError(f"{sym!r} is used after {sym._consumed}.")
        if consume:
            sym._consumed = consume
        return sym

    ############################################################################
    # Demuxing
    ############################################################################
    def open(
        self,
        src: _Symbol,
        demux_config: object,
        name: str | None,
    ) -> _Symbol:
        self.use(src, "source")
        self.opened.add(src._reg)
        out = self._new("demuxer", cls=_DemuxerSymbol)
        self._emit(
            "open",
            [src._reg],
            [out._reg],
            src=src._reg,
            out=out._reg,
            demux_config=demux_config,
            name=name,
        )
        return out

    def demux(
        self,
        demuxer: _Symbol,
        media: str,
        window: tuple[tuple[int, int], tuple[int, int]] | None,
        bsf: str | None,
    ) -> _Symbol:
        self.use(demuxer, "demuxer")
        # The default filter trims with `packets.timestamp`, which is the window
        # after the native rational reduction. Compute it the same way.
        timestamp = (
            None
            if window is None
            else _libspdl.FusedPlan.window_to_timestamp(  # pyrefly: ignore[missing-attribute]
                window
            )
        )
        out = self._new("packets", media, timestamp)
        self._emit(
            "demux",
            [demuxer._reg],
            [out._reg],
            media=media,
            src=demuxer._reg,
            out=out._reg,
            window=window,
            bsf=bsf,
        )
        return out

    def bsf(self, packets: _Symbol, bsf: str) -> _Symbol:
        self.use(packets, "packets", consume="bit stream filtering")
        out = self._new("packets", packets._media, packets._timestamp)
        self._emit(
            "bsf",
            [packets._reg],
            [out._reg],
            media=packets._media,
            src=packets._reg,
            out=out._reg,
            bsf=bsf,
        )
        return out

    ############################################################################
    # Decoding
    ############################################################################
    def decode(
        self,
        packets: _Symbol,
        resolve_filter: Callable[[str, tuple[float, float] | None], str | None],
        decode_config: object,
        num_frames: int,
    ) -> _Symbol:
        self.use(packets, "packets", consume="decoding")
        media = packets._media
        assert media is not None
        out = self._new("frames", media)
        self._emit(
            "decode",
            [packets._reg],
            [out._reg],
            media=media,
            src=packets._reg,
            out=out._reg,
            filter_desc=resolve_filter(media, packets._timestamp),
            decode_config=decode_config,
            num_frames=num_frames,
        )
        return out

    def sample_decode(
        self,
        packets: _Symbol,
        indices: list[int],
        filter_desc: str | None,
        decode_config: object,
        optimized: bool,
    ) -> list[_Symbol]:
        self.use(packets, "packets", media="video", consume="sample decoding")
        outs = [self._new("frames", "image") for _ in indices]
        self._emit(
            "sample_decode",
            [packets._reg],
            [o._reg for o in outs],
            src=packets._reg,
            outs=[o._reg for o in outs],
            indices=list(indices),
            filter_desc=filter_desc,
            decode_config=decode_config,
            optimized=optimized,
        )
        return outs

    ############################################################################
    # Conversion
    ############################################################################
    def convert(self, frames: object, storage: object) -> _Symbol:
        batch = not isinstance(frames, _Symbol)
        items: list[object] = (
            [frames] if isinstance(frames, _Symbol) else list(_as_sequence(frames))
        )
        if not items:
            raise _UsageValueError("`frames` must not be empty.")
        syms = [self.use(f, "frames") for f in items]
        if len({s._media for s in syms}) != 1:
            raise _UsageTypeError("All the frames must be the same media type.")
        out = self._new("buffer")
        regs = [s._reg for s in syms]
        self._emit(
            "convert",
            regs,
            [out._reg],
            media=syms[0]._media,
            srcs=regs,
            out=out._reg,
            batch=batch,
            storage=storage,
        )
        return out

    def external(
        self,
        sym: _Symbol,
        op: object,
        kind: str,
        media: str | None,
        consume: str,
    ) -> _Symbol:
        self.use(sym, kind, media=media, consume=consume)
        out = self._new("cuda_buffer")
        self._emit(
            "external",
            [sym._reg],
            [out._reg],
            op=op,
            srcs=[sym._reg],
            out=out._reg,
        )
        return out

    def post(self, buffer: _Symbol, fn: Callable[[Any], Any]) -> _Symbol:
        self.use(buffer, "buffer", "cuda_buffer", consume=f"`{fn.__name__}`")
        return _PostSymbol(buffer, fn)

    ############################################################################
    # Metadata, clone and slice
    ############################################################################
    def attr(self, sym: _Symbol, name: str) -> Any:
        if self.is_materialized(sym):
            return getattr(self.concrete(sym), name)
        if name == "clone" and sym._kind in ("packets", "frames"):
            return functools.partial(self.clone, sym)
        spec = self._attr_specs.get((sym._kind, sym._media), {}).get(name)
        if spec is None:
            if sym._kind not in _MOVED_KINDS:
                raise AttributeError(f"{sym!r} has no attribute `{name}`.")
            # Not recordable. Execute up to here and use the actual value.
            return getattr(self.concrete(sym), name)
        form, kind, media = spec
        if form == _PROP:
            return self.query(sym, name, kind, media)

        def method(*, raw: bool = False) -> _Symbol:
            return self.query(sym, name, kind, media, raw=raw)

        return method

    def query(
        self,
        sym: _Symbol,
        name: str,
        kind: str,
        media: str | None,
        raw: bool = False,
    ) -> _Symbol:
        self.use(sym, sym._kind)
        out = self._new(kind, media)
        self._emit(
            "query",
            [sym._reg],
            [out._reg],
            src=sym._reg,
            out=out._reg,
            name=name,
            raw=raw,
        )
        return out

    def clone(self, sym: _Symbol) -> _Symbol:
        self.use(sym, "packets", "frames")
        out = self._new(sym._kind, sym._media, sym._timestamp)
        self._emit("clone", [sym._reg], [out._reg], src=sym._reg, out=out._reg)
        return out

    def slice(self, sym: _Symbol, key: object) -> Any:
        if self.is_materialized(sym):
            return self.concrete(sym)[key]
        self.use(sym, "frames", media="video")
        regs = {"src": sym._reg}
        if isinstance(key, int) and not isinstance(key, bool):
            out = self._new("frames", "image")
            self._emit(
                "slice_index", [sym._reg], [out._reg], out=out._reg, index=key, **regs
            )
            return out
        out = self._new("frames", "video")
        if isinstance(key, slice):
            parts = (key.start, key.stop, key.step)
            if not all(v is None or _is_int(v) for v in parts):
                raise TypeError(f"Slice must consist of integers. Found: {key}")
            start, stop, step = parts
            self._emit(
                "slice_range",
                [sym._reg],
                [out._reg],
                out=out._reg,
                start=start,
                stop=stop,
                step=step,
                **regs,
            )
            return out
        if isinstance(key, Sequence) and all(_is_int(v) for v in key):
            self._emit(
                "slice_indices",
                [sym._reg],
                [out._reg],
                out=out._reg,
                indices=list(key),
                **regs,
            )
            return out
        raise TypeError(
            "Traced frames can be indexed with an integer, a slice or "
            f"a list of integers known at trace time. Found: {key!r}"
        )


class _LazyGraph(_Graph):
    """Executes recorded operations whenever Python code needs their results."""

    def __init__(self, state: Any) -> None:
        super().__init__()
        self._state = state
        self._flushed = 0
        self.num_executions = 0
        # Executions to convert buffers to arrays (`to_numpy` etc.).
        # A traced program records them as the conversion of the outputs, so
        # they do not prevent tracing with `cache=True`.
        self.num_post_executions = 0
        self.values: dict[int, Any] = {}
        self._pending_meta: list[int] = []
        # Sources bound to the native registers, which must outlive them.
        self._sources: list[object] = []
        # Registers whose values are released.
        self._dropped: set[int] = set()

    def bind_source(self, sym: _Symbol, src: object) -> None:
        self._state.bind([sym._reg], [src])
        self._sources.append(src)

    def open_source(
        self, src: object, demux_config: object, name: str | None
    ) -> _Symbol:
        resolved = _resolve_src(src)
        sym = self._new("source")
        self.bind_source(sym, resolved)
        demuxer = self.open(sym, demux_config, name)
        assert isinstance(demuxer, _DemuxerSymbol)
        demuxer._source = (resolved, demux_config, name)
        return demuxer

    def streaming_demux(
        self, demuxer: _Symbol, indices: set[int], num_packets: int, duration: float
    ) -> Any:
        assert isinstance(demuxer, _DemuxerSymbol)
        self.use(demuxer, "demuxer")
        if demuxer._source is None:
            raise NotImplementedError("The source of the demuxer is not available.")
        if demuxer._eager is None:
            src, demux_config, name = demuxer._source
            demuxer._eager = (
                _libspdl.make_demuxer(  # pyrefly: ignore[missing-attribute]
                    src, demux_config=demux_config, name=name
                )
            )
        return demuxer._eager.streaming_demux(
            {_actual(i) for i in indices}, num_packets=num_packets, duration=duration
        )

    def is_materialized(self, sym: _Symbol) -> bool:
        return sym._kind in _MOVED_KINDS and sym._reg in self.values

    def _new(
        self,
        kind: str,
        media: str | None = None,
        timestamp: tuple[float, float] | None = None,
        cls: type[_Symbol] | None = None,
    ) -> _Symbol:
        sym = super()._new(kind, media, timestamp, cls)
        if kind == "meta":
            self._pending_meta.append(sym._reg)
        return sym

    def flush(self, exports: list[int]) -> list[Any]:
        if not exports and self._flushed == len(self.ops):
            return []
        return self._execute(exports)

    def _execute(self, exports: list[int]) -> list[Any]:
        self.num_executions += 1
        try:
            plan = self._build(exports)
        except Exception as e:
            raise _FusionError(e) from e
        try:
            return plan.run_on(self._state, exports)
        except Exception:
            self.failed = True
            raise

    def _build(self, exports: list[int]) -> Any:
        """Build the native plan of the operations recorded since the last execution."""
        plan = _libspdl.FusedPlan(  # pyrefly: ignore[missing-attribute]
            self.num_registers, [], []
        )
        for op, kwargs, _, _ in self.ops[self._flushed :]:
            getattr(plan, f"add_{op}")(**kwargs)
        self._flushed = len(self.ops)
        if dead := self._dead_registers(exports):
            # Release the values in the same native call, without the GIL.
            plan.add_drop(dead)
            self._dropped.update(dead)
        return plan

    def _dead_registers(self, exports: list[int]) -> list[int]:
        """Registers that no operation recorded from now on can use.

        These are the registers of the symbols that are garbage collected or
        consumed (e.g. a closed demuxer). Releasing them after the pending
        operations is safe, since operations are recorded through symbols.
        """
        keep = set(exports) | self._dropped
        return [
            reg
            for reg, ref in self.symbols.items()
            if reg not in keep and ((sym := ref()) is None or sym._consumed)
        ]

    def concrete(self, sym: _Symbol) -> Any:
        if sym._reg not in self.values:
            self.materialize([sym])
        return self.values[sym._reg]

    def materialize(self, syms: list[_Symbol]) -> None:
        """Execute pending operations, and fetch the values of ``syms``.

        Pending metadata queries are fetched together.
        """
        for sym in syms:
            self.use(sym, *_OUTPUT_KINDS)
        exports = [r for r in self._pending_meta if r not in self.values]
        exports += [s._reg for s in syms if s._reg not in exports]
        self._pending_meta = []
        for reg, value in zip(exports, self.flush(exports), strict=True):
            self.values[reg] = value

    def finish(self, result: object) -> object:
        """Execute the remaining operations and return the result of the function.

        The values in lists, tuples, namedtuples, dicts and in the fields of
        dataclass instances are replaced with the actual values. Other symbols
        that are still referenced (e.g. as attributes of other objects) are
        materialized, so that they keep working.
        """
        leaves: list[Any] = []
        tree = _flatten(result, leaves)
        pending: dict[int, _Symbol] = {}
        for o in leaves:
            if isinstance(o, _Symbol) and o._reg not in self.values:
                if o._graph is not self:
                    raise ValueError(f"{o!r} belongs to a different fused function.")
                pending[o._reg] = o
        for ref in self.symbols.values():
            sym = ref()
            if (
                sym is not None
                and sym._reg not in pending
                and sym._reg not in self.values
                and sym._kind in _OUTPUT_KINDS
                and not sym._consumed
            ):
                pending[sym._reg] = sym
        self.materialize(list(pending.values()))
        visited: set[int] = set()
        results = [self._replace(o, visited) for o in leaves]
        return _unflatten(tree, iter(results))

    def _replace(self, obj: object, visited: set[int]) -> object:
        """Replace placeholders with their actual values.

        In dataclass instances, the fields are replaced in place, so the
        instance keeps its identity (and the same instance referenced from
        elsewhere sees the actual values).
        """
        if isinstance(obj, _Symbol):
            return self.values[obj._reg] if obj._reg in self.values else obj
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            if id(obj) not in visited:
                visited.add(id(obj))
                for field in dataclasses.fields(obj):
                    value = getattr(obj, field.name)
                    replaced = self._replace_tree(value, visited)
                    if replaced is not value:
                        # `object.__setattr__` also works for frozen dataclasses.
                        object.__setattr__(obj, field.name, replaced)
            return obj
        return obj

    def _replace_tree(self, obj: object, visited: set[int]) -> object:
        leaves: list[Any] = []
        tree = _flatten(obj, leaves)
        if tree is None:
            return self._replace(obj, visited)
        replaced = [self._replace(o, visited) for o in leaves]
        if all(r is o for r, o in zip(replaced, leaves, strict=True)):
            return obj
        return _unflatten(tree, iter(replaced))

    def post(self, buffer: _Symbol, fn: Callable[[Any], Any]) -> Any:
        before = self.num_executions
        value = self.concrete(buffer)
        self.num_post_executions += self.num_executions - before
        return fn(value)


def _is_int(v: object) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _as_sequence(obj: object) -> Sequence[object]:
    if not isinstance(obj, Sequence):
        raise _UsageTypeError(
            f"Expected frames or a sequence of frames. Found: {type(obj)}"
        )
    return obj


################################################################################
# Entry points for the public functions when they receive a traced value
################################################################################


_ACTIVE: contextvars.ContextVar["_Graph | None"] = contextvars.ContextVar(
    "spdl_io_fuse_active_graph", default=None
)


def _active_graph() -> "_Graph | None":
    """The graph recording the fused function running in this context."""
    return _ACTIVE.get()


@contextlib.contextmanager
def _suspend() -> Iterator[None]:
    """Execute SPDL IO functions eagerly, even in a fused function."""
    token = _ACTIVE.set(None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


@contextlib.contextmanager
def _activate(g: "_Graph") -> Iterator[None]:
    token = _ACTIVE.set(g)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _is_symbol(obj: object) -> TypeGuard[_Symbol]:
    """Whether ``obj`` is a recorded value whose operations can be recorded."""
    return isinstance(obj, _Symbol) and not obj._graph.is_materialized(obj)


def _unwrap_materialized(obj: T) -> T:
    """Replace a symbol, which Python already needed, with its value."""
    if isinstance(obj, _Symbol) and obj._graph.is_materialized(obj):
        return obj._graph.concrete(obj)
    return obj


def _unwrap_frames(
    frames: "_Frames | Sequence[_Frames]",
) -> "_Frames | Sequence[_Frames]":
    """:py:func:`_unwrap_materialized` for (a sequence of) frames.

    If only a part of the frames are materialized, all of them are.
    """
    if isinstance(frames, _Symbol) or not isinstance(frames, Sequence):
        return _unwrap_materialized(frames)
    syms = [f for f in frames if isinstance(f, _Symbol)]
    if syms and not all(_is_symbol(f) for f in syms):
        return [_unwrap(f) for f in frames]  # type: ignore[misc]
    return frames


def _actual(obj: T, *, unrecorded: bool = False) -> T:
    """The actual value of ``obj``, for operations that cannot be recorded.

    ``unrecorded`` stops cache tracing before an eager operation can consume a
    concrete non-source argument. Lazy execution still uses the eager path.
    """
    if (
        unrecorded
        and (g := _ACTIVE.get()) is not None
        and not isinstance(g, _LazyGraph)
    ):
        raise _GraphBreak("an operation on a concrete input that cannot be recorded")
    return obj._graph.concrete(obj) if isinstance(obj, _Symbol) else obj


@_recording
def _record_open_source(
    src: object, *, demux_config: object, name: str | None, kwargs: dict[str, Any]
) -> Any:
    """Record opening a concrete source, if a fused function is running."""
    if kwargs or (g := _ACTIVE.get()) is None:
        return None
    return g.open_source(src, demux_config, name)


def _fallback(feature: str) -> NoReturn:
    """Stop tracing, as ``feature`` needs the actual values."""
    raise _GraphBreak(feature)


def _is_traced(obj: object) -> bool:
    if _is_symbol(obj):
        return True
    return isinstance(obj, Sequence) and any(_is_symbol(o) for o in obj)


def _packets_info(packets: _Symbol) -> tuple[str, tuple[float, float] | None]:
    """The media type and the timestamp of traced packets, known while tracing."""
    packets._graph.use(packets, "packets")
    assert packets._media is not None
    return packets._media, packets._timestamp


@_recording
def _record_open(
    src: _Symbol, *, demux_config: object, name: str | None, kwargs: dict[str, Any]
) -> Any:
    if kwargs:
        raise TypeError(f"Unsupported arguments in fused function: {sorted(kwargs)}")
    return src._graph.open(src, demux_config, name)


@_recording
def _record_bsf(packets: _Symbol, bsf: str) -> Any:
    return packets._graph.bsf(packets, bsf)


@_recording
def _record_decode(
    packets: _Symbol,
    resolve_filter: Callable[[str, tuple[float, float] | None], str | None],
    *,
    decode_config: object,
    num_frames: int,
) -> Any:
    return packets._graph.decode(packets, resolve_filter, decode_config, num_frames)


@_recording
def _record_sample_decode(
    packets: _Symbol,
    indices: list[int],
    *,
    filter_desc: str | None,
    decode_config: object,
    optimized: bool,
) -> Any:
    return packets._graph.sample_decode(
        packets, indices, filter_desc, decode_config, optimized
    )


@_recording
def _record_convert(frames: object, storage: object) -> Any:
    first = frames if isinstance(frames, _Symbol) else _as_sequence(frames)[0]
    if not isinstance(first, _Symbol):
        raise TypeError("Traced and concrete frames cannot be mixed.")
    return first._graph.convert(frames, storage)


@_recording
def _record_cuda(
    sym: _Symbol,
    op: object,
    kind: str,
    *,
    media: str | None = None,
    consume: str = "",
) -> Any:
    """Record an operation supplied by the optional CUDA extension."""
    return sym._graph.external(sym, op, kind, media, consume)


@_recording
def _record_post(buffer: _Symbol, fn: Callable[[Any], Any]) -> Any:
    return buffer._graph.post(buffer, fn)


################################################################################
# Argument handling
################################################################################

# A tree definition is `None` for a leaf, `(list|tuple, children)` or
# `(dict, keys, children)`.
_TreeDef = Any


def _flatten(obj: object, leaves: list[Any]) -> _TreeDef:
    """Flatten lists, tuples, namedtuples and dicts into ``leaves``.

    Only these exact types are flattened, and they are rebuilt as the same
    type. Other objects, including subclasses of these types, are leaves, so
    that they are not rebuilt as a different type.
    """
    if isinstance(obj, (list, tuple)) and (
        type(obj) in (list, tuple) or _is_namedtuple(type(obj))
    ):
        return (type(obj), tuple(_flatten(o, leaves) for o in obj))
    if isinstance(obj, dict) and type(obj) is dict:
        keys = tuple(obj)
        return (dict, keys, tuple(_flatten(obj[k], leaves) for k in keys))
    leaves.append(obj)
    return None


def _is_namedtuple(typ: type) -> bool:
    return hasattr(typ, "_fields")


def _unflatten(tree: _TreeDef, leaves: Any) -> Any:
    if tree is None:
        return next(leaves)
    if tree[0] is dict:
        _, keys, children = tree
        return {k: _unflatten(c, leaves) for k, c in zip(keys, children, strict=True)}
    typ, children = tree
    values = [_unflatten(c, leaves) for c in children]
    return typ(*values) if _is_namedtuple(typ) else typ(values)


def _is_source(obj: object) -> bool:
    if isinstance(obj, (str, bytes, bytearray, memoryview, os.PathLike)):
        return True
    if hasattr(obj, "numpy"):  # torch tensor
        return True
    try:
        memoryview(obj)  # type: ignore[arg-type]
    except TypeError:
        return False
    return True


def _const_key(obj: object) -> object:
    try:
        hash(obj)
    except TypeError as e:
        raise TypeError(
            f"Non-source arguments of a fused function must be hashable. Found: {type(obj)}"
        ) from e
    return (type(obj), obj)


################################################################################
# Tracing
################################################################################


_LAZY = object()
_TRACE = object()
_CACHE_SIZE = 128

# How each output of a fused function is produced: from an output of the
# plan (optionally converted by a function), or a constant.
_OutSlot = tuple[str, Any]


class _Program:
    def __init__(self, plan: Any, slots: list[_OutSlot], out_tree: _TreeDef) -> None:
        self.plan = plan
        self.slots = slots
        self.out_tree = out_tree

    def run(self, srcs: list[Any]) -> Any:
        values = self.plan.run(srcs)
        results = []
        for kind, value in self.slots:
            if kind == "const":
                results.append(value)
            elif kind == "same":
                results.append(results[value])
            else:
                i, fn = value
                results.append(values[i] if fn is None else fn(values[i]))
        return _unflatten(self.out_tree, iter(results))


def _check_output(g: _Graph, o: object) -> _Symbol:
    if not isinstance(o, _Symbol) or o._graph is not g:
        raise TypeError(
            "A fused function must return values produced by SPDL IO functions, "
            f"constants, or a list/tuple/dict of them. Found: {o!r}"
        )
    return g.use(o, *_OUTPUT_KINDS)


def _resolve_outputs(g: _Graph, outputs: list[Any]) -> tuple[list[int], list[_OutSlot]]:
    """Map the leaves of the result to the outputs of the plan.

    An object that appears more than once in the result is produced once,
    and the same object is returned at each position, as without ``fuse``.
    """
    regs: list[int] = []
    slots: list[_OutSlot] = []
    first: dict[int, int] = {}
    for o in outputs:
        if isinstance(o, _CONSTANT_TYPES):
            slots.append(("const", o))
        elif not isinstance(o, _Symbol):
            # A Python object, which must be created on every call.
            raise _GraphBreak(f"returning {type(o)}")
        elif id(o) in first:
            slots.append(("same", first[id(o)]))
        else:
            first[id(o)] = len(slots)
            if isinstance(o, _PostSymbol):
                reg, fn = o._base._reg, o._fn
            else:
                reg, fn = _check_output(g, o)._reg, None
            if reg in regs:
                # E.g. a buffer and its conversion: the buffer is consumed.
                raise _GraphBreak("returning a value and its conversion")
            regs.append(reg)
            slots.append(("out", (len(regs) - 1, fn)))
    return regs, slots


def _build_plan(g: _Graph, out_regs: list[int]) -> Any:
    last_use: dict[int, int] = {}
    for i, (_, _, uses, defs) in enumerate(g.ops):
        for r in uses + defs:
            last_use[r] = i
    keep = set(out_regs) | {s._reg for s in g.inputs}

    plan = _libspdl.FusedPlan(  # pyrefly: ignore[missing-attribute]
        g.num_registers, [s._reg for s in g.inputs], out_regs
    )
    for i, (op, kwargs, uses, defs) in enumerate(g.ops):
        getattr(plan, f"add_{op}")(**kwargs)
        if dead := sorted(
            {r for r in uses + defs if last_use[r] == i and r not in keep}
        ):
            plan.add_drop(dead)
    return plan


def _make_args(
    g: _Graph, tree: _TreeDef, leaves: list[Any], is_src: list[bool]
) -> tuple[list[Any], dict[str, Any]]:
    traced = [
        g.source() if src else leaf for leaf, src in zip(leaves, is_src, strict=True)
    ]
    return _unflatten(tree, iter(traced))


def _trace(
    fn: Callable[..., Any], tree: _TreeDef, leaves: list[Any], is_src: list[bool]
) -> _Program:
    g = _Graph()
    args, kwargs = _make_args(g, tree, leaves, is_src)
    with _activate(g):
        result = fn(*args, **kwargs)
    if unused := [s for s in g.inputs if s._reg not in g.opened]:
        # The argument might not be a source. Its value must not be ignored.
        raise _GraphBreak(f"{unused} are not used as sources of SPDL IO.")
    outputs: list[Any] = []
    out_tree = _flatten(result, outputs)
    out_regs, slots = _resolve_outputs(g, outputs)
    return _Program(_build_plan(g, out_regs), slots, out_tree)


def _run(
    fn: Callable[..., T], args: Any, kwargs: Any, switch: "_Switch"
) -> tuple[T, bool | None]:
    """Run ``fn`` while recording SPDL IO functions.

    If executing the recorded operations fails, ``fn`` is executed again
    without recording, so that the error is raised where the SPDL IO function
    is called, and the function handles it as it does without ``fuse``.

    If recording fails (fusion, as opposed to an operation, fails), fusion
    is disabled for the function with ``switch``, and ``fn`` is executed
    again without recording.

    Also returns whether the recorded operations were executed as one
    native call at the end. (``None`` when it is not known.)
    """
    state = _libspdl.FusedState()  # pyrefly: ignore[missing-attribute]
    g = _LazyGraph(state)
    try:
        with _activate(g):
            result = fn(*args, **kwargs)
            straight = g.num_executions == g.num_post_executions
            try:
                result = g.finish(result)  # type: ignore[assignment]
            except Exception as e:
                if g.failed:
                    raise
                raise _FusionError(e) from e
        if not g.failed:
            return result, straight  # pyrefly: ignore[bad-return]
    except _FusionError as e:
        switch.disable(e.error)
    except Exception:
        if not g.failed:
            raise
    finally:
        state.close()
    return fn(*args, **kwargs), None


def _gil_enabled() -> bool:
    """Whether the GIL is enabled.

    Checked on every call, because free-threaded Python re-enables the GIL
    when it imports an extension module that does not support running
    without it.
    """
    is_gil_enabled = getattr(sys, "_is_gil_enabled", None)
    return True if is_gil_enabled is None else is_gil_enabled()


def _try_trace(
    fn: Callable[..., Any], tree: _TreeDef, leaves: list[Any], is_src: list[bool]
) -> object:
    """Trace ``fn``, or return ``_LAZY`` if it cannot be traced.

    Any error while tracing means that the function cannot be traced (e.g.
    an argument classified as a source is not a source), so the function
    runs without the cache. A genuine error is raised again when it runs.
    """
    try:
        return _trace(fn, tree, leaves, is_src)
    except (_GraphBreak, Exception):
        return _LAZY


def _qualified_name(fn: Callable[..., Any]) -> str:
    module = getattr(fn, "__module__", None) or "<unknown>"
    qualname = getattr(fn, "__qualname__", None) or repr(fn)
    return f"{module}.{qualname}"


class _Switch:
    """Whether fusion is disabled for a fused function.

    Initialized with the switch read when the function is decorated. Fusion
    is also disabled, for the rest of the process, when recording a call of
    the function fails.
    """

    __slots__ = ("name", "disabled")

    def __init__(self, name: str) -> None:
        self.name = name
        self.disabled: bool = _is_fusion_disabled(name)

    def disable(self, error: Exception) -> None:
        if not self.disabled:
            self.disabled = True
            _LG.warning(
                "Fusion failed for `%s`, so it runs without fusion from now on.",
                self.name,
                exc_info=error,
            )


def _run_eagerly(switch: _Switch) -> bool:
    """Whether a fused function must run as if it was not decorated."""
    return switch.disabled or _ACTIVE.get() is not None or not _gil_enabled()


def _cached(fn: Callable[P, T]) -> Callable[P, T]:
    # Argument structure -> how to run. `_TRACE` means trace on the next call.
    cache: OrderedDict[object, object] = OrderedDict()
    switch = _Switch(_qualified_name(fn))

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if _run_eagerly(switch):
            return fn(*args, **kwargs)
        leaves: list[Any] = []
        tree = _flatten((args, kwargs), leaves)
        is_src = [_is_source(leaf) for leaf in leaves]
        try:
            key = (
                tree,
                tuple(
                    _SOURCE_KEY if src else _const_key(leaf)
                    for leaf, src in zip(leaves, is_src, strict=True)
                ),
            )
        except TypeError:
            return _run(fn, args, kwargs, switch)[0]
        program = cache.get(key)
        if program is _TRACE:
            program = _try_trace(fn, tree, leaves, is_src)
            cache[key] = program
        if isinstance(program, _Program):
            cache.move_to_end(key)
            srcs = [
                _resolve_src(leaf)
                for leaf, src in zip(leaves, is_src, strict=True)
                if src
            ]
            try:
                return program.run(srcs)
            except Exception:
                # Raise and handle the error as without `fuse`.
                return fn(*args, **kwargs)
        result, straight = _run(fn, args, kwargs, switch)
        if program is None and straight is not None:
            cache[key] = _TRACE if straight else _LAZY
            if len(cache) > _CACHE_SIZE:
                cache.popitem(last=False)
        return result

    return wrapper


def _fuse(fn: Callable[P, T]) -> Callable[P, T]:
    switch = _Switch(_qualified_name(fn))

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        if _run_eagerly(switch):
            return fn(*args, **kwargs)
        return _run(fn, args, kwargs, switch)[0]

    return wrapper


@overload
def fuse(fn: Callable[P, T], /, *, cache: bool = False) -> Callable[P, T]: ...
@overload
def fuse(*, cache: bool = False) -> Callable[[Callable[P, T]], Callable[P, T]]: ...


def fuse(
    fn: Callable[P, T] | None = None, /, *, cache: bool = False
) -> Callable[P, T] | Callable[[Callable[P, T]], Callable[P, T]]:
    """**[Experimental]** Execute SPDL IO functions in a function with fewer native calls.

    Each SPDL IO function call crosses the boundary between Python and C++,
    and releases and re-acquires the GIL. When a sample is loaded with
    multiple calls (demuxing, decoding and conversion, possibly repeated in
    a loop), re-acquiring the GIL can wait for other Python threads, which
    dominates the loading time in a busy process. ``fuse`` records the
    SPDL IO function calls in the decorated function, and executes them in C++
    with one boundary crossing when the function returns.

    The function runs as usual, except that SPDL IO functions return
    placeholders of their results (packets, frames, buffers and their
    metadata). When Python code needs an actual value (e.g.
    ``if frames.num_frames == 0``, a filter computed from ``packets.width``,
    or tensor operations on the result of :py:func:`~spdl.io.to_torch`), the
    operations recorded so far are executed in one native call. Operations
    that cannot be recorded (e.g. streaming demuxing and decoding) run as
    usual with the actual values. The placeholders
    in the returned value (in lists, tuples, namedtuples, dicts and fields of
    dataclass instances) are replaced with the actual values.

    The recorded functions are :py:class:`~spdl.io.Demuxer`,
    :py:func:`~spdl.io.demux_audio`, :py:func:`~spdl.io.demux_video`,
    :py:func:`~spdl.io.demux_image`, :py:func:`~spdl.io.apply_bsf`,
    :py:func:`~spdl.io.decode_packets`,
    :py:func:`~spdl.io.sample_decode_video`,
    :py:func:`~spdl.io.convert_frames`, :py:func:`~spdl.io.load_audio`,
    :py:func:`~spdl.io.load_video`, :py:func:`~spdl.io.load_image`,
    :py:func:`~spdl.io.load_image_batch`, :py:func:`~spdl.io.get_filter_desc`,
    :py:func:`~spdl.io.transfer_buffer`, :py:func:`~spdl.io.decode_packets_nvdec`,
    :py:func:`~spdl.io.nv12_to_rgb` and :py:func:`~spdl.io.nv12_to_bgr`.
    CUDA operations are recorded when their input is recorded and the CUDA
    extension supports fusion. CUDA buffer attributes materialize the buffer.
    Python allocator callbacks still acquire the GIL when allocating or freeing.
    Metadata of demuxers, codecs, packets and frames, ``clone()`` and
    indexing of video frames are recorded as well.

    **When the GIL is disabled (free-threaded Python), fuse does nothing**,
    and the function runs eagerly. Without the GIL, a native call does not
    wait for other threads, so recording does not save time.

    If executing the recorded operations fails (e.g. a source is corrupted),
    the function is executed again without recording, so that the error is
    raised where the SPDL IO function is called, and the function handles it
    (or not) as it does without ``fuse``.

    If recording fails instead (fusion fails, e.g. the function uses SPDL IO
    in a way that ``fuse`` does not support), the function is executed again
    without recording, and fusion is disabled for the function for the rest
    of the process. A warning with the cause is logged once.

    .. note::

       In both cases, Python code in the function runs twice. Avoid side
       effects that must not be repeated, such as consuming an iterator or
       modifying an argument.

    The following figure and table show the time per call of decoding 32 PNG
    images and batching them, demuxing 4 windows of a video and decoding them,
    :py:func:`~spdl.io.load_audio`, and a function that computes the filter
    from ``packets.width``, which cannot be cached, as GIL contention grows
    from 0% to 100% (Python 3.14 and its free-threaded build 3.14t, mean and
    95% confidence interval of 10 trials, in milliseconds). GIL contention is
    the share of time another Python thread runs Python code. Without fusion,
    every native call has to re-acquire the GIL, so the time grows quickly with
    the contention. With fusion, a call re-acquires the GIL about once, so its
    time grows much more slowly. Compared to the gain of fusing itself, the
    difference that ``cache=True`` makes is marginal. Without the GIL, the time
    does not change. The dotted line in the figure is the time without
    contention, and its y-axis is in log scale.

    .. image:: /_static/data/fuse_benchmark.png
       :alt: Time per call of the functions without fusion, with fuse, with
             fuse(cache=True), and without fusion on the free-threaded
             build, as GIL contention grows from 0% to 100%.

    =============  ==========  ==============  ============  ============  ===========
    Function       Contention  Eager           fuse          cache=True    3.14t eager
    =============  ==========  ==============  ============  ============  ===========
    32 PNG images  0%          5.05 ± 0.03     4.59 ± 0.02   3.92 ± 0.02   5.20 ± 0.06
    32 PNG images  10%         5.59 ± 0.03     4.79 ± 0.12   4.03 ± 0.02   5.14 ± 0.02
    32 PNG images  25%         6.95 ± 0.06     5.00 ± 0.04   5.00 ± 0.03   5.17 ± 0.06
    32 PNG images  50%         10.65 ± 0.12    5.36 ± 0.33   5.01 ± 0.02   5.18 ± 0.06
    32 PNG images  75%         22.34 ± 0.53    9.87 ± 0.16   7.98 ± 0.09   5.20 ± 0.05
    32 PNG images  90%         63.56 ± 3.25    9.88 ± 0.13   8.51 ± 0.13   5.16 ± 0.06
    32 PNG images  100%        839.74 ± 24.37  10.03 ± 0.04  9.32 ± 0.03   5.10 ± 0.02
    video windows  0%          3.39 ± 0.01     3.46 ± 0.02   3.24 ± 0.02   3.46 ± 0.01
    video windows  10%         3.64 ± 0.03     3.48 ± 0.02   3.29 ± 0.04   3.45 ± 0.01
    video windows  25%         4.54 ± 0.07     3.69 ± 0.10   3.34 ± 0.02   3.48 ± 0.03
    video windows  50%         6.93 ± 0.08     5.05 ± 0.03   5.02 ± 0.02   3.48 ± 0.04
    video windows  75%         12.44 ± 0.46    7.40 ± 0.11   6.76 ± 0.12   3.44 ± 0.03
    video windows  90%         29.96 ± 1.71    7.60 ± 0.10   7.12 ± 0.23   3.42 ± 0.03
    video windows  100%        81.04 ± 4.39    8.91 ± 0.03   8.63 ± 0.05   3.42 ± 0.03
    load_audio     0%          1.62 ± 0.01     1.72 ± 0.01   1.59 ± 0.01   1.64 ± 0.01
    load_audio     10%         1.71 ± 0.03     1.75 ± 0.03   1.65 ± 0.01   1.64 ± 0.01
    load_audio     25%         2.11 ± 0.08     2.05 ± 0.05   2.01 ± 0.03   1.65 ± 0.02
    load_audio     50%         3.29 ± 0.11     3.40 ± 0.06   2.99 ± 0.18   1.64 ± 0.01
    load_audio     75%         6.55 ± 0.05     3.92 ± 0.07   3.50 ± 0.12   1.63 ± 0.01
    load_audio     90%         16.04 ± 1.49    5.03 ± 0.00   5.03 ± 0.00   1.66 ± 0.02
    load_audio     100%        36.86 ± 1.63    7.15 ± 0.08   6.98 ± 0.06   1.63 ± 0.01
    metadata used  0%          2.45 ± 0.01     2.61 ± 0.01   2.61 ± 0.01   2.48 ± 0.02
    metadata used  10%         2.52 ± 0.03     2.70 ± 0.02   2.71 ± 0.02   2.47 ± 0.04
    metadata used  25%         3.08 ± 0.07     3.36 ± 0.05   3.35 ± 0.05   2.46 ± 0.03
    metadata used  50%         4.91 ± 0.10     5.09 ± 0.08   5.04 ± 0.02   2.46 ± 0.02
    metadata used  75%         9.62 ± 0.22     10.01 ± 0.04  10.04 ± 0.03  2.46 ± 0.02
    metadata used  90%         15.33 ± 2.09    10.07 ± 0.00  10.07 ± 0.00  2.46 ± 0.02
    metadata used  100%        33.72 ± 0.98    18.69 ± 0.09  18.67 ± 0.07  2.47 ± 0.04
    =============  ==========  ==============  ============  ============  ===========

    See :doc:`../io/fusion` for the details and guidelines.

    .. versionadded:: 0.9.0

    Args:
        fn: A function that loads media with SPDL IO functions.
        cache: *Optional:* If ``True``, after a call executed the recorded
            operations as one native call, the next call with the same
            argument structure records the program by replacing the source
            arguments (``str``, ``bytes`` and other buffer objects) with
            placeholders. The program is used for the subsequent calls with
            the same argument structure, without executing the function.
            The argument structure is the nesting of lists, tuples,
            namedtuples and dicts (exact types), and the values of non-source
            arguments. The programs for the last
            128 argument structures are kept. This saves the cost of
            executing the function in Python, but it is only correct if
            the function depends only on its arguments. (If an argument
            classified as a source is used in other ways, e.g. formatted
            into a string, the function is not cached.)

            Compared to the gain of fusing itself, the difference that
            ``cache=True`` makes is marginal (see the table above). At 100% GIL
            contention, fusing saves 15 to 830 ms per call, and the cache saves
            a further 0.2 to 0.7 ms. Without contention, the cache saves 0.1 to
            0.7 ms.

    Returns:
        A function with the same signature as ``fn``.
        (If ``fn`` is omitted, a decorator.)

    .. admonition:: Example

       .. code-block::

          @spdl.io.fuse
          def load_batch(srcs: list[bytes]) -> spdl.io.CPUBuffer:
              frames = [
                  spdl.io.decode_packets(spdl.io.demux_image(src))
                  for src in srcs
              ]
              return spdl.io.convert_frames(frames)

          buffer = load_batch([data1, data2])  # One call into C++
    """
    wrap = _cached if cache else _fuse
    return wrap if fn is None else wrap(fn)  # pyrefly: ignore[bad-return]
