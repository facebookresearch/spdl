# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This module provides building block definitions for :py:class:`~spdl.pipeline.Pipeline`.

.. seealso::

   :ref:`Example: Pipeline definitions <example-pipeline-definitions>`
      Illustrates how to build a complex pipeline.

You can build a pipeline by creating a :py:class:`PipelineConfig`,
then passing it to :py:func:`spdl.pipeline.build_pipeline` function.

The following step explains the steps to build a pipeline using the building blocks.

- Instantiate a :py:class:`SourceConfig` by providing a ``Iterable`` or
  ``AsyncIterable``.
- Using :py:func:`Pipe`, :py:func:`Aggregate` or :py:func:`Disaggregate`,
  instantiate a series of :py:class:`PipeConfig`.
- Instantiate a :py:class:`SinkConfig`.
- Pass the config objects created in the previous steps to :py:class:`PipelineConfig`.
- Pass the ``PipelineConfig`` object to :py:func:`spdl.pipeline.build_pipeline` function.
"""

from ._defs import (
    Aggregate,
    Disaggregate,
    Merge,
    MergeConfig,
    Pipe,
    PipeConfig,
    PipelineConfig,
    SinkConfig,
    SourceConfig,
)

__all__ = [
    "Aggregate",
    "Disaggregate",
    "Merge",
    "MergeConfig",
    "Pipe",
    "PipeConfig",
    "PipelineConfig",
    "SinkConfig",
    "SourceConfig",
]
