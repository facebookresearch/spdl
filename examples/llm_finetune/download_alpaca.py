# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Download the Stanford Alpaca dataset and save as JSONL.

Source: https://github.com/tatsu-lab/stanford_alpaca
License: CC BY-NC 4.0
~52K instruction-following samples with instruction/input/output fields.

Usage:
  buck run //spdl/examples/llm_finetune:download_alpaca -- \
    --output /tmp/alpaca.jsonl
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
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
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

    _LG.info("Downloading Alpaca dataset from %s", args.url)
    with urllib.request.urlopen(args.url) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    _LG.info("Downloaded %d samples", len(data))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for sample in data:
            row = {
                "instruction": sample["instruction"],
                "input": sample.get("input", ""),
                "output": sample["output"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    _LG.info("Saved to %s", args.output)


if __name__ == "__main__":
    sys.exit(main())
