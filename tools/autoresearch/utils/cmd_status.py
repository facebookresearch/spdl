# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .log import setup_logging
from .state import read_config, read_master_table, read_state

_LG: logging.Logger = logging.getLogger(__name__)


def _parse_args(args: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show experiment status.")
    parser.add_argument("workdir")
    return parser.parse_args(args)


def run(args: list[str]) -> None:
    ns = _parse_args(args)
    workdir = Path(ns.workdir).resolve()
    setup_logging(workdir)

    state = read_state(workdir)
    config = read_config(workdir)

    max_iter = config["stopping_criteria"]["max_iterations"]
    print(f"Experiment : {workdir}")
    print(f"Status     : {state['status']}")
    print(f"Iteration  : {state['iteration']}/{max_iter}")
    print(f"Baseline   : {state.get('baseline_job', 'none')}")
    print(f"Best run   : {state.get('current_best', 'none')}")
    print(f"Image      : {state.get('cached_image', 'none')}")
    print(f"\nMaster table:\n{read_master_table(workdir)}")
