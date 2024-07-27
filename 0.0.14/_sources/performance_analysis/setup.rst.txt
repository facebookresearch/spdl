Building a Pipeline
===================

.. py:currentmodule:: spdl.pipeline

Now, we use :py:class:`Pipeline` to construct the data loader.
When training models in the cloud, the process of making data
available on GPUs typically involves 4 to 5 steps.

#. Download raw data
#. Load and preprocess data
#. Batch data
#. Preprocess the batch (optional)
#. Transfer the batch to GPUs

When using SPDL's pipeline abstraction, this process can be
written as follows.

.. code-block:: python

   from spdl.pipeline import PipelineBuilder

   # List of URLs
   class Dataset(Iterable[str]):
       ...

   # Download the data
   def download(urls: list[str]) -> list[bytes]:
       ...

   # Load tensor from the raw data with some additional preprocessing
   def preprocess(data: bytes) -> Tensor:
       ...

   # Create a batch Tensor
   def collate(samples: list[Tensor]) -> Tensor:
       ...

   # Transfer the batch tensor to the GPU
   def gpu_transfer(batch: Tensor) -> Tensor:
       ...

   # Build Pipeline
   pipeline = (
       PipelineBuilder()
       .add_source(Dataset())
       .aggregate(...)
       .pipe(download, concurrency=...)
       .disaggregate()
       .pipe(preprocess, concurrency=...)
       .aggregate(batch_size)
       .pipe(collate)
       .pipe(gpu_transfer)
       .add_sink()
       .build(num_threads=...)
   )

   # Run
   with pipeline.auto_stop():
       for batch in pipeline.get_iterator(timeout):
           ...

Typically, network calls are more efficient when requests are batched,
so we aggregate the URLs before making a network call and then
disaggregate them afterward.

Since downloading takes some time but does not consume many CPU resources,
we make multiple download calls concurrently.

Decoding the raw data and applying preprocessing can be time-consuming and
computationally intensive.
As `previously described <./noisy_neighbour.html>`_,
it is recommended to keep total CPU utilization at around 40% to avoid a QPS drop.
However, we want to prevent the training process from suffering from
data starvation.
For this purpose, we apply some concurrency to the preprocessing stage.

The data are then batched and transferred to the GPU.
These stages typically do not require concurrency.
Concurrent GPU data transfer is not feasible due to hardware constraints.
