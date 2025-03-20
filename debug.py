import logging
logging.basicConfig(level=logging.DEBUG)
from spdl.io.lib import _libspdl_cuda, _libspdl

import torch

# torch.zeros([1]).to('cuda')
_libspdl_cuda.init

print('done', flush=True)
