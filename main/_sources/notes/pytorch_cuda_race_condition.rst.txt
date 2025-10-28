PyTorch CUDA Race Condition in Multi-threading
==============================================

.. py:currentmodule:: spdl.io

When using PyTorch with multi-threading and multiple CUDA streams for data transfer,
a subtle race condition can occur that affects the reliability and correctness of tensor operations.

Background
----------

The :py:func:`transfer_tensor` function is a handy function that encapsulates the steps required to
transfer data to the GPU without interfering with the main thread and the default CUDA stream.

It does the following:

1. Create a thread-local CUDA stream.
2. Activate the thread-local CUDA stream.
3. Move the data to page-locked memory (pin-memory).
4. Instruct asynchronous data transfer to the GPU (``cudaMemcpyAsync()``).
5. Synchronize the stream to ensure that the transfer is completed.

The function is intended to be executed in a background thread, so the above steps
allow continuous streaming of data to the GPU.

Typically with :py:class:`~spdl.pipeline.Pipeline`, the pipeline is set up as follows:

.. code-block::

   pipeline = (
      PipelineBuilder()
      .add_source(dataset)
      .pipe(preprocess)
      .pipe(  # Background data transfer
          transfer_tensor,
          executor=io_executor,
      )
      .add_sink()
      .build(...)
   )

   with pipeline.auto_stop():
       for batch in pipeline:
           model(batch)  # foreground model computation


.. mermaid::

   flowchart LR
       subgraph Box1[Background Thread]
           subgraph s1[Custom CUDA Stream]
               item1["Transfer"]
           end
       end

       subgraph Box2[Foreground Thread]
           subgraph s2[Default CUDA Stream]
               item2["Model Computation"]
           end
       end

       item1 e1@--> item2
       e1@{ animation: slow }

The Problem
-----------

We encountered a case where certain models cause data corruption.
We checked the data at each step of the pipeline and noticed that occasionally the
data the model is processing does not match the data sent to the GPU.

The first step of the model was as follows:

.. code-block::

   img = img.float() / 255

This is a typical normalization of image data.
This works fine for CPU, but for GPU there are two issues.

When Python finishes executing this line, the following things happen:

- The reference to the original data and intermediate data (float conversion) are lost.
- The actual computation (CUDA kernels) is queued, but not necessarily executed or completed.

Most of the time, it still functions fine, but when multi-threading and multiple CUDA streams
are involved, things go wrong in subtle ways.

PyTorch has a CUDA Caching Allocator (CCA), which allocates a large chunk of GPU memory
and uses it in segments as requested by PyTorch operations.
When all the Storages that point to a particular buffer are deleted,
the CCA reuses the buffer for future use.
When managing CUDA memory segments, the CCA records the CUDA stream that was active at the
time the buffer was requested, and uses it to infer if the memory is still being used.
The CCA will delay the deallocation (and reuse) until the associated stream is synchronized.
For normal model computation, this ensures that the memory segments are active while
forward/backward paths are being executed.

However, if the first allocation happened in a non-default stream, the CCA will not consider
the default stream as the context being used.
So, when a reference to a CUDA tensor is lost in a stream other than its origin stream,
the CCA can immediately reuse the underlying memory region.
However, when the reference to the CUDA tensor is lost in Python code, the corresponding
CUDA kernels are scheduled but might not yet be executed.

When CUDA tensors are created in a background thread, right after a reference to an
existing CUDA tensor is lost but before the CUDA kernel is executed, the background thread
can create another CUDA tensor, and the CCA can assign the same memory region to the new tensor.

The result is data corruption. The following diagram illustrates this:

.. image:: ../../_static/data/pytorch_cuda_race_condition.png


The Solution: Caching the reference in ``transfer_tensor``
----------------------------------------------------------

PyTorch provides a mitigation for multi-stream applications, such as
:py:meth:`~torch.Tensor.record_stream`.
However, we are not certain that this works for multi-threading situations,
so we resort to a more primitive approach.

The :py:func:`transfer_tensor` function now holds a strong reference to the batches
transferred to the GPU.
It holds references to the last 4 batches it transferred.
4 is the number of batches that can exist in the current pipeline stage and
the next pipeline stage.
The pipeline stages are connected with queues and each queue can hold up to 2 items.

.. mermaid::

   flowchart LR
   subgraph UpstreamBox[Transfer Stage]
       item4["Batch 4"]
   end

   subgraph Queue[Queue]
       item2["Batch 3"]
       item3["Batch 2"]
   end

   subgraph DownstreamBox[Model Stage]
       item1["Batch 1"]
   end

   UpstreamBox e1@--> Queue
   Queue e2@--> DownstreamBox

   e1@{ animation: slow }
   e2@{ animation: slow }

By holding the last 4 batches, even if the reference is lost in the model stage,
the last reference is still held by the cache of :py:func:`transfer_tensor`, so
the CCA will not immediately reuse the underlying memory.


References
----------

1. `CUDA Caching Allocator <https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html>`_
2. `PyTorch CUDA Streams Documentation <https://docs.pytorch.org/docs/2.9/notes/cuda.html#cuda-streams>`_
3. `torch.Tensor.record_stream <https://docs.pytorch.org/docs/2.9/generated/torch.Tensor.record_stream.html>`_
4. `FSDP CUDACachingAllocator Discussion <https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486>`_
