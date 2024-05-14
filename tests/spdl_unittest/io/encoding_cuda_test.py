import asyncio
from itertools import product
from tempfile import NamedTemporaryFile

import pytest

import spdl.io

import torch
from spdl.io import encode_config

# Mostly smoke test
# TODO: Inspect the output

_DEFAULT_DEVICE = torch.device("cuda:7")


@pytest.mark.parametrize(
    "fmt,enc_cfg",
    product(
        [((16, 16, 3), "rgb24"), ((16, 16), "gray"), ((3, 16, 16), "yuv444p")],
        [None, encode_config(width=128, height=96, scale_algo="neighbor")],
    ),
)
def test_encode_smoketest(fmt, enc_cfg):
    shape, pix_fmt = fmt
    data = torch.randint(255, size=shape, dtype=torch.uint8, device=_DEFAULT_DEVICE)
    print(data, flush=True)

    async def _test(arr):
        with NamedTemporaryFile(suffix=".png") as f:
            await spdl.io.async_encode_image(
                f.name,
                arr,
                pix_fmt=pix_fmt,
                encode_config=enc_cfg,
            )

    asyncio.run(_test(data))
