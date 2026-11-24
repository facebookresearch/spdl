.. _building-inference-pipeline:

Building an inference pipeline
==============================

.. admonition:: tl;dr
   :class: tip

   SPDL can pipeline the entire inference workflow, from fetching inputs to
   saving results. Overlapping I/O, CPU work, and GPU computation across batches
   can reduce the time the GPU spends waiting.

SPDL was originally designed to address data-loading bottlenecks in model
training. Its stages can also span an entire inference job. A training loader
usually feeds a loop that runs forward and backward passes and updates the
model. In an inference job, the forward pass can be a stage in the middle of the
pipeline, followed by stages that move results back to the CPU, post-process
them, and write them to an external system.

.. mermaid::

   flowchart TD
       A[Fetch data from remote storage] --> B[Load data and preprocess]
       B --> C[Transfer to GPU]
       C --> D[Run inference]
       D --> E[Transfer results to CPU]
       E --> F[Post-process and serialize]
       F --> G[Send results to an external system]

Challenge: Keeping GPU Busy
---------------------------

In a sequential loop, each iteration waits for input preparation, the model,
transfers, and result writes. For example, with ``device`` set to a CUDA device:

.. code-block:: python

   with torch.inference_mode():
       for batch in data_loader:
           batch = batch.to(device)
           result = model(batch)
           result = result.to("cpu")
           save_result(result)

While the CPU prepares inputs or saves results, no other batch is submitted to
the model. The baseline plot shows the gaps in GPU activity in this workload.

.. raw:: html

   <div id="baseline_smu"></div>

Solution: Software Pipelining with SPDL
---------------------------------------

Split the workflow into stages connected by bounded queues. The sketch below
uses placeholders for the application's stage functions and source:

.. code-block:: python

   from spdl.pipeline import PipelineBuilder

   pipeline = (
       PipelineBuilder()
       .add_source(source)
       .pipe(get_data_from_remote)
       .pipe(preprocess)
       .pipe(send_to_gpu)
       .pipe(inference)
       .pipe(send_to_cpu)
       .pipe(put_data_to_remote)
       .add_sink(2)
       .build(num_threads=6)
   )

SPDL coordinates the queues on its event loop and dispatches synchronous stage
functions to worker threads. While the GPU processes one batch, the pipeline
can fetch and preprocess the next and save the previous batch's results. The
following diagram shows that overlap across batches:

.. image:: /_static/data/inference_pipelining.png
   :alt: Fetch, model execution, and result handling overlap across batches

.. raw:: html

   <div id="spdl_smu"></div>

The pipelined run reached 96% SM utilization and stayed consistent after
warm-up in this comparison. Utilization describes device activity, not
completed work or savings. Compare equivalent inputs and output writes,
including startup and the final drain, before claiming a throughput or cost
improvement.

Running GPU Inference in Background Thread
------------------------------------------

Putting GPU computation in the middle of the pipeline lets the application
schedule both input work and expensive work that follows the model, such as
converting tensors to images and uploading them. This is the main difference
from a loader that only supplies batches to a foreground training loop.

With PyTorch, select the CUDA device explicitly and enter
``torch.inference_mode()`` in the inference stage; inference mode is
thread-local. When different threads or streams handle producer and consumer
work, order transfers using CUDA events. For example,
:py:func:`spdl.io.transfer_tensor_d2h` accepts a ``producer_event`` recorded on
the thread and stream that produced the CUDA tensors. It waits for the copy to
complete before returning CPU output. Bound result buffering: retained results
from that helper occupy pinned host memory.

.. include:: ../../plots/inference.txt
