Building Inference Pipeline
===========================

.. admonition:: tl;dr
   :class: tip

   SPDL can be used to pipeline the entire inference workflow.
   Using SPDL to manage all I/O, CPU, and GPU operations,
   you can maximize GPU efficiency and reduce GPU hours.

In this section, we explain how to run an inference pipeline with SPDL.

The primary goal of the SPDL project is to solve the data loading bottleneck in model training.
We designed SPDL's Pipeline to be flexible and generic to support various data flow patterns.
It turned out that this generic design can naturally support inference tasks.

Inference pipelines involve loading data onto the GPU, running inference,
moving the result from the GPU to the CPU, and saving the result.

.. mermaid::

   flowchart TD
       A[Fetch data from remote storage] --> B[Load data and preprocess]
       B --> C[Transfer to GPU]
       C --> D[Perform Inference]
       D --> E[Transfer the result to CPU]
       E --> F[Post-process and serialize]
       F --> G[Send the result to external system]

The differences between training and inference are as follows:

- In training, the model computation involves forward/backward passes
  and parameter updates, and they are carried out in the foreground (main) thread.
- In inference, the model computation only performs the forward pass.
  When running the inference with SPDL, the model computation
  (and data transfer between CPU and GPU) is performed in background threads.
- In inference, the result of the model computation is sent back to the CPU and then
  further sent to an external system.

Challenge: Keeping GPU Busy
----------------------------

In a naive sequential approach, the GPU sits idle while data is being
transferred between the CPU and the GPU, and results are being saved:

.. code-block:: python

   for batch in data_loader:
       batch = batch.to(GPU)    # GPU idle
       result = model(batch)    # GPU busy
       result = result.to(CPU)  # GPU idle
       save_result(result)      # GPU idle

The following figure illustrates how the GPU utilization (SM utilization)
would look like.


.. raw:: html

   <div id="baseline_smu"></div>

The baseline shows inconsistent GPU utilization with significant idle time.

Solution: Software Pipelining with SPDL
----------------------------------------

SPDL solves this by splitting the workflow into stages and executing them asynchronously.
Each stage runs in parallel, connected by queues that allow data to flow continuously
through the pipeline.

.. code-block:: python

   pipeline = (
       PipelineBuilder()
       .add_source(source)
       .pipe(get_data_from_remote)
       .pipe(preprocess)
       .pipe(send_to_gpu)
       .pipe(inference)
       .pipe(send_to_cpu)
       .pipe(put_data_to_remote)
       .add_sink(...)
   )

SPDL converts each function into an asynchronous task and orchestrates their
execution using an asyncio event loop. While one batch is being processed by the GPU,
the next batch is being fetched and preprocessed, and the previous batch's results
are being saved. This creates a continuous flow that keeps the GPU busy.

.. image:: ../../_static/data/inference_pipelining.png

The following figure

.. raw:: html

   <div id="spdl_smu"></div>

With SPDL's pipelined approach, SM utilization reaches 96% and stays consistent
after warm-up, representing a significant improvement over the baseline.

.. include:: ../plots/inference.txt

Running GPU Inference in Background Thread
-------------------------------------------

One novel advancement demonstrated in this work is **performing GPU computation
in the middle of the pipeline**. Previously, SPDL was primarily used to handle
input operations building up to a GPU operation that ran on the main thread
(typical in training workloads).

In inference applications, there is often substantial CPU and I/O intensive work
to do *after* the GPU call (such as converting tensors to images and uploading
results). By having SPDL manage the GPU inference in the middle of a longer pipeline,
all I/O, CPU, and GPU workloads in the entire inference flow are globally optimized
by SPDL, maximally leveraging the hardware.

The key insight is that SPDL's GIL-free multi-threading and queue-based
architecture are perfectly suited for end-to-end inference workflows,
enabling seamless integration of:

- Data loading/saving
- Data processing
- CPU-GPU memory transfer
- CUDA kernel calls
