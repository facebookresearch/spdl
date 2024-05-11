# Buffer Object

::: spdl.io
    options:
      heading_level: 2
      show_signature_annotations: true
      show_bases: true
      show_root_heading: false
      show_root_toc_entry: false
      members:
      - CPUBuffer
      - CUDABuffer

## Cast buffer to array/tensor

Buffer objects can be cast into array/tensor object.
Currently, SPDL supports, NumPy, PyTorch and Numba.

Casting to respective framework requires these framework
to be installed.

::: spdl.io.to_numpy
::: spdl.io.to_torch
::: spdl.io.to_numba
