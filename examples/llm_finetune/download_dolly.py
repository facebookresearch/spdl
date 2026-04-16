# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Download the Databricks Dolly 15K dataset and save as JSONL.

Source: https://huggingface.co/datasets/databricks/databricks-dolly-15k
License: CC BY-SA 3.0
~15K instruction-following samples across 8 task categories.

Usage:
  buck run //spdl/examples/llm_finetune:download_dolly -- \
    --output /tmp/dolly.jsonl
"""

# pyre-strict

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

_LG: logging.Logger = logging.getLogger(__name__)

_URL: str = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k"
    "/resolve/main/databricks-dolly-15k.jsonl"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Output JSONL file path"
    )
    parser.add_argument("--url", type=str, default=_URL, help="URL to download from")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    _LG.info("Downloading Dolly 15K dataset from %s", args.url)
    with urllib.request.urlopen(args.url) as resp:
        raw = resp.read().decode("utf-8")

    # Dolly fields: instruction, context, response, category
    # Normalize to Alpaca format: instruction, input, output
    count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            row = {
                "instruction": sample["instruction"],
                "input": sample.get("context", ""),
                "output": sample["response"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    _LG.info("Saved %d samples to %s", count, args.output)


if __name__ == "__main__":
    sys.exit(main())
