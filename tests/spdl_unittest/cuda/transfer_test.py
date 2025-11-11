# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from spdl.io import transfer_tensor


def test_gpu_transfer() -> None:
    ref = torch.randint(256, (16, 3, 4608, 5328), dtype=torch.uint8)
    print(ref)

    cuda = transfer_tensor(ref)
    print(cuda)
    assert cuda.device.type == "cuda"
    assert torch.equal(cuda, ref.cuda())
