# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""This example shows how to build :py:class:`~spdl.pipeline.Pipeline`
object with Hydra using building blocks from :py:mod:`spdl.pipeline.defs`
module.

The definition of the pipeline is found in ``"hydra_integration.yaml"`` file.

.. literalinclude:: ../../../examples/hydra_integration.yaml
    :language: yaml

"""

__all__ = ["main"]

import os

import hydra
from omegaconf import DictConfig

os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base=None, config_path=".", config_name="hydra_integration")
def main(cfg: DictConfig):
    """The main entry point.

    Args:
        cfg: The configuration created from the ``"hydra_integration.yaml"`` file.
    """
    pipeline_cfg = hydra.utils.instantiate(cfg.pipeline_cfg)
    print(pipeline_cfg)

    pipeline = hydra.utils.instantiate(cfg.pipeline)
    print(pipeline)

    with pipeline.auto_stop():
        for i, item in enumerate(pipeline.get_iterator(timeout=3)):
            print(i, f"{item=}")


if __name__ == "__main__":
    main()
