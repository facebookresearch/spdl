# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded ordinary-Torch GPU correctness checks for the standalone H2D helper."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

import spdl.io._transfer as module
import torch
from spdl.io import transfer_tensor_h2d


class _LaterFailure(dict):
    def __init__(self, *args, **kwargs):
        if args:
            raise ValueError("later mapping field failed")
        super().__init__(**kwargs)


def _assert_independent() -> None:
    forbidden = [name for name in sys.modules if name.startswith("spdl.pipeline")]
    forbidden += [name for name in sys.modules if name == "spdl.io.lib." + "_buffer"]
    if forbidden:
        raise AssertionError(
            "Standalone validation loaded pipeline or new buffer modules"
        )


def _delay_cycles(stream: torch.cuda.Stream) -> int:
    # A bounded delay creates the pending-work window under test; it is not an
    # idle GPU task or a utilization policy workaround.
    sample = 10_000_000
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    with torch.cuda.stream(stream):
        begin.record()
        torch.cuda._sleep(sample)
        end.record()
    end.synchronize()
    seconds = max(begin.elapsed_time(end) / 1000, 0.000001)
    return min(2_000_000_000, max(sample, int(sample * 0.75 / seconds)))


def _full_values() -> dict:
    source = torch.arange(262144, dtype=torch.float32, pin_memory=True)
    output = transfer_tensor_h2d(source, num_caches=0)
    if not torch.equal(output.cpu(), source):
        raise AssertionError("Entire ordinary pinned H2D value mismatch")
    return {
        "case": "ordinary_pinned_full_values",
        "elements": source.numel(),
        "passed": True,
    }


def _late_traversal_error() -> dict:
    source = torch.arange(262144, dtype=torch.float32, pin_memory=True)
    try:
        transfer_tensor_h2d({"first": source, "later": _LaterFailure()}, num_caches=0)
    except ValueError as error:
        if str(error) != "later mapping field failed":
            raise
    else:
        raise AssertionError("Expected the primary traversal error")
    helper = module._get_transfer_func(0)
    if helper._pending is not None or not helper._stream.query():
        raise AssertionError(
            "Earlier ordinary copies were not drained after traversal failure"
        )
    return {"case": "partial_traversal_drains_real_submitted_copy", "passed": True}


def _default_stream_lifetime(num_caches: int) -> dict:
    # Give each cache configuration its own standard thread-local helper.
    module._THREAD_LOCAL = threading.local()
    stream = torch.cuda.default_stream(0)
    # Warm the real operations outside the checked pressure window.
    warm = transfer_tensor_h2d(
        torch.ones(262144, pin_memory=True), num_caches=num_caches
    )
    warm.mul(2).add(1)
    torch.cuda.synchronize()
    del warm
    cycles = _delay_cycles(stream)
    helper = module._get_transfer_func(num_caches)
    outputs, expected, completions, pressure_receipts = [], [], [], []
    count = max(6, num_caches + 3)
    for sequence in range(count):
        source = torch.arange(262144, dtype=torch.float32, pin_memory=True).add_(
            sequence
        )
        gpu = transfer_tensor_h2d(source, num_caches=num_caches)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(cycles)
            outputs.append(gpu.mul(2).add(sequence))
            completion = torch.cuda.Event()
            completion.record(stream)
        expected.append(source.mul(2).add(sequence))
        completions.append(completion)
        del source, gpu
        gc.collect()
        # Pressure the original allocation stream after the corresponding GPU
        # input is evicted; no caller record_stream or global synchronization.
        evicted = sequence - num_caches
        if evicted < 0:
            continue
        pending_before = not completions[evicted].query()
        with torch.cuda.stream(helper._stream):
            pressure = [
                torch.empty(262144, dtype=torch.float32, device="cuda")
                for _ in range(8)
            ]
            for allocation in pressure:
                allocation.fill_(-17)
        helper._stream.synchronize()
        pending_after = not completions[evicted].query()
        pressure_receipts.append(
            {
                "sequence": evicted,
                "pending_before": pending_before,
                "pending_after": pending_after,
            }
        )
        del pressure
    stream.synchronize()
    for actual, target in zip(outputs, expected, strict=True):
        if not torch.equal(actual.cpu(), target):
            raise AssertionError(
                "Whole default-stream value mismatch under allocator pressure"
            )
    overlaps = sum(
        row["pending_before"] and row["pending_after"] for row in pressure_receipts
    )
    if overlaps == 0:
        raise AssertionError(
            "No evicted input had pending default-stream work during pressure"
        )
    return {
        "case": "default_stream_destination_lifetime",
        "num_caches": num_caches,
        "checked_batches": count,
        "pending_pressure_events": overlaps,
        "pressure": pressure_receipts,
        "passed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-sha", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    assert module.__file__ is not None
    digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if digest != args.expected_source_sha:
        raise RuntimeError(
            "The loaded H2D source does not match the requested immutable source"
        )
    _assert_independent()
    os.environ["LOCAL_RANK"] = "0"
    torch.cuda.set_device(0)
    cases = [
        _full_values(),
        _late_traversal_error(),
        _default_stream_lifetime(0),
        _default_stream_lifetime(4),
    ]
    torch.cuda.synchronize()
    _assert_independent()
    report = {
        "status": "passed",
        "source_sha256": digest,
        "source_file": module.__file__,
        "torch": torch.__version__,
        "cases": cases,
        "scope": (
            "Four bounded ordinary-Torch correctness cases; no new buffer/pipeline "
            "modules, driver sync poisoning or performance sweep."
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"status": "passed", "source_sha256": digest, "cases": len(cases)})
    )


if __name__ == "__main__":
    main()
