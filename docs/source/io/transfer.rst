CPU/GPU Data Transfer
=====================

In the context of ML/AI application efficiency, it is utmost importance to
keep the GPU running.
To do so, the input data must be transfered to GPU memory without interrupting
the model computation.
Similarly in inference, the result of the computation must be sent back to CPU
memory without interrupting the model computation.

The following diagram illustrates this.

.. mermaid::

   flowchart LR
    subgraph B[GPU]

    Compute e1@ --> Compute

    end
    A([Input Batch]) e2@--> B e3@--> C([Result Tensor])

    e1@{animate: true}
    e2@{animate: true}
    e3@{animate: true}

CPU to GPU transfer
-------------------

Conceptually this idea is simple, however, achieving this is more complicated
than one imagines.

1. Model computation and data transfer must use different CUDA streams.
   (CPU to GPU and GPU to CPU transfer can share the stream if they are fast enough, which is typically the case.)
2. To asynchrnously copy data from CPU to GPU, the data must be first written to a page-locked (a.k.a. pinned) memory.
3. If more than one Tensor needs to be transfered, one can use asynchronous copy, such as ``cudaMemcpyAsync`` then synchronize the CUDA stream.

In PyTorch, to achieve the above, couple of APIs must be used properly in multi-threaded setting.

1. Create a dedicated CUDA stream with :py:class:`torch.cuda.Stream`
2. Activate the CUDA stream with :py:func:`torch.cuda.stream` context manager.
3. Call :py:meth:`~torch.Tensor.pin_memory` on the batch tensor.
4. Transfer the data using :py:meth:`~torch.Tensor.to` with ``non_blocking=True``.
5. Synchronize the CUDA stream with :py:meth:`torch.cuda.Stream.synchronize`.

Although this is doumented in `the Developer Note <https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-streams>`_,
we found that many practitioners are confused and not properly using the APIs.
So SPDL provides a function that makes this process easy.

The :py:func:`spdl.io.transfer_tensor` function manages a thread-local CUDA stream
and transfer data asynchronously in the CUDA stream.
You can check out the documentation and its implementation.
When using this function, it is recommended to use a dedicated thread.

You can achieve this with :py:class:`~spdl.pipeline.Pipeline` as follow.

.. seealso::

   :ref:`pipeline-parallelism-custom-mt`

.. code-block:: python

   transfer_executor = ThreadPoolExecutor(max_workers=1)

   pipeline = (
       PipelineBuilder()
       .add_source(...)
       .pipe(...)
       .pipe(transfer_data, executor=transfer_executor)
       .add_sink(...)
   )

This way, the transfer function is always executed in a dedicated thread, so that
it keeps using the same CUDA stream.

When tracing this pipeline with
`PyTorch Profiler <https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_,
we can see that it is always the one background thread that issues data transfer,
and the transfer overlaps with the stream executing the model training.

.. image:: ../../_static/data/parallelism_transfer.png
   

GPU to CPU transfer
-------------------

The above concept applies to the GPU to CPU transfer as well.
However, as far as we know, PyTorch does not provide a way to
copy data from GPU memory to page-locked CPU memory.

You can apply the same technique, but the transfer from GPU to CPU
is not truly async.

We are working to provide a solution for this.
