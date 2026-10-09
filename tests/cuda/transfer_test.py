# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
from spdl.io import transfer_tensor_h2d


class TransferTest(unittest.TestCase):
    def test_gpu_transfer(self) -> None:
        """The named H2D API transfers a representative tensor to CUDA."""
        ref = torch.randint(256, (16, 3, 4608, 5328), dtype=torch.uint8)
        print(ref)

        cuda = transfer_tensor_h2d(ref)
        print(cuda)
        self.assertEqual(cuda.device.type, "cuda")
        torch.testing.assert_close(cuda, ref.cuda())
