Overview of pipeline
====================

When training models in the cloud, the process to make data
available on GPUs takes typically 4 ~ 5 steps.

#. Download raw data
#. Load and preprocess data
#. Batch data
#. Preprocess the batch (optional)
#. Transfer the batch to GPUs

When using SPDL's pipeline abstraction, such process can be
written as follow.

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


Typically, network calls can be more efficient when requests are batched,
so we aggregate the URLs before making a network call, then disaggregate
after that.

Since downloading takes some time but does not consume much CPU resources,
we make multiple download calls concurrently.

Decoding the raw data and applying preprocessing can be time-consuming and
compute-intensive. In AI model training, it is recommended to keep the total
CPU utilization at most around 40% for the sake of keeping training QPS high.
(When CPU utilization is high, CPU cannot schedule GPU kernel launch in
timely manner. Every CPU utilization comes with drop in training QPS.)
However, we want to prevent the training from suffering from data starvation.
For this purpose, we want to apply some concurrency to preprocessing stage.

The data are then batched and transfered to GPU. These stages usually
do not require concurrency. It is not feasible to make the GPU data transfer
concurrent due to the hardware constraints.
