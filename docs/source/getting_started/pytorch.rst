Building High-Performance DataLoaders
=====================================

The :py:class:`~spdl.pipeline.Pipeline` operates on iterables and applies a series of functions to process data efficiently. While the ``Pipeline`` class implements the ``Iterable`` protocol and can be used directly in ``for batch in pipeline:`` loops, many ML practitioners prefer a PyTorch-style DataLoader interface that they are familiar with.

This guide explains how to build high-performance data loading solutions using SPDL pipelines, and highlights the key conceptual differences from PyTorch's approach.

Understanding the Paradigm Shift
---------------------------------

The fundamental difference between SPDL and PyTorch's data loading approach lies in how data processing is structured and executed.

PyTorch DataLoader Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In PyTorch, the typical data loading pattern uses three components:

1. **Sampler** - Generates indices for data access
2. **Dataset** - Maps an index to processed data (typically a Tensor)
3. **DataLoader** - Handles batching, multiprocessing, and prefetching

The following diagram illustrates the PyTorch DataLoader structure:

.. mermaid::

   flowchart TB
       subgraph P[PyTorch DataLoader]
           direction TB
           Sampler --> DataSet[Dataset: Map int → Tensor]
           DataSet --> Buffer[Prefetch Buffer]
       end

In this model, the Dataset encapsulates all data processing logic—from loading raw data to producing final Tensors. This monolithic approach has limitations:

- All processing happens in a single ``__getitem__`` call
- Different types of operations (I/O, CPU processing, memory operations) are not distinguished
- Limited opportunities for fine-grained concurrency control
- Difficult to optimize individual processing stages

SPDL Pipeline Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~

SPDL takes a different approach by decomposing data processing into discrete stages:

.. mermaid::

   flowchart TB
       subgraph S[SPDL Pipeline]
           direction TB
           src[Source: Iterator]
           subgraph stages[Processing Stages]
               direction TB
               p1[Stage 1: I/O Operations<br/>e.g., Load from storage]
               p2[Stage 2: Decoding<br/>e.g., Decode image/video]
               p3[Stage 3: Preprocessing<br/>e.g., Resize, normalize]
               p1 --> p2 --> p3
           end
           src --> stages
           stages --> sink[Sink Buffer]
       end

This decomposition enables:

- **Separation of concerns**: Different operations can be configured independently
- **Optimized concurrency**: I/O-bound and CPU-bound operations can use different execution strategies
- **Better resource utilization**: Each stage can be tuned for its specific workload
- **Reduced memory overhead**: Data stays in efficient formats until batching

The Key Insight: Split I/O and CPU Work
----------------------------------------

To achieve high performance with SPDL, you must split your data loading logic into stages based on their resource requirements:

**I/O-Bound Operations** (Network, Disk)
   - Fetching data from remote storage
   - Reading files from disk
   - Database queries
   - Best executed with high concurrency (many threads or async operations)

**CPU-Bound Operations** (Computation)
   - Image/video decoding
   - Data transformations
   - Preprocessing and augmentation
   - Best executed with moderate concurrency (matching CPU cores)

**Memory Operations** (Data Movement)
   - Batching individual items
   - Converting to contiguous memory
   - Device transfers (CPU → GPU)
   - Often best executed serially or with low concurrency

Restructuring PyTorch Datasets for SPDL
----------------------------------------

Consider a typical PyTorch Dataset:

.. code-block:: python

   class MyDataset:
       def __init__(self, data_urls: list[str]):
           self.data_urls = data_urls

       def __len__(self) -> int:
           return len(self.data_urls)

       def __getitem__(self, index: int) -> torch.Tensor:
           # Everything happens here - I/O, decoding, preprocessing
           url = self.data_urls[index]
           raw_data = download(url)           # I/O-bound
           decoded = decode_image(raw_data)   # CPU-bound
           processed = resize_image(decoded)  # CPU-bound
           return torch.tensor(processed)     # Memory operation

To use this with SPDL effectively, decompose it into a pipeline builder function:

.. code-block:: python

   from typing import Any, Callable
   from collections.abc import Iterable
   from spdl.pipeline import PipelineBuilder
   from spdl.source import DistributedRandomSampler
   import spdl.io
   import torch
   import numpy as np

   def build_data_pipeline(
       catalog: list[str],
       sampler: Iterable[int],
       *,
       load_fn: Callable[[int], Any],
       preprocess_fn: Callable[[Any], Any] | None = None,
       collate_fn: Callable[[list[Any]], torch.Tensor],
       batch_size: int,
       drop_last: bool = False,
       num_io_workers: int = 16,
       num_cpu_workers: int = 8,
       buffer_size: int = 10,
   ) -> PipelineBuilder:
       """Build a pipeline that processes data in stages.

       Args:
           catalog: List of data identifiers (e.g., URLs, file paths, database keys)
           sampler: An iterable that yields indices (e.g., DistributedRandomSampler)
           load_fn: Function to load data given an index
           preprocess_fn: Optional function to preprocess loaded data
           collate_fn: Function to collate a list of items into a batch
           batch_size: Number of items per batch
           drop_last: Whether to drop the last incomplete batch
           num_io_workers: Concurrency level for I/O operations
           num_cpu_workers: Concurrency level for CPU processing
           buffer_size: Size of the prefetch buffer

       Returns:
           A configured PipelineBuilder ready to build pipelines
       """
       # Build the pipeline with the provided sampler
       builder = (
           PipelineBuilder()
           .add_source(sampler)

           # Stage 1: I/O operations (high concurrency)
           .pipe(
               lambda idx: load_fn(catalog[idx]),
               concurrency=num_io_workers,
               output_order="completion",  # Don't wait for slow requests
           )
       )

       # Stage 2: CPU processing (moderate concurrency, if provided)
       if preprocess_fn is not None:
           builder.pipe(
               preprocess_fn,
               concurrency=num_cpu_workers,
               output_order="input",  # Maintain order
           )

       # Stage 3: Batching and collation
       builder.aggregate(batch_size, drop_last=drop_last)
       builder.pipe(collate_fn)

       # Stage 4: GPU transfer (serial execution)
       builder.pipe(spdl.io.transfer_tensor)

       # Prefetch buffer
       builder.add_sink(buffer_size)

       return builder

.. note::

   The :py:func:`spdl.io.transfer_tensor` function combines and encapsulates multiple operations required to transfer data from CPU to GPU in the background without interrupting model computation in the default CUDA stream. This includes the "pin memory" operation, which moves data to page-locked memory regions for faster transfer. Unlike PyTorch's DataLoader, there is no separate ``pin_memory`` parameter—this optimization is built into ``transfer_tensor``.

Now we can use this function to implement the equivalent of ``MyDataset``:

.. code-block:: python

   # Define the processing functions that match MyDataset's logic
   def load_data(url: str) -> bytes:
       return download(url)  # I/O-bound

   def process_data(raw_data: bytes) -> np.ndarray:
       decoded = decode_image(raw_data)    # CPU-bound
       processed = resize_image(decoded)   # CPU-bound
       return processed

   def collate_batch(items: list[np.ndarray]) -> torch.Tensor:
       return torch.stack([torch.tensor(item) for item in items])

   # Build the pipeline with a sampler
   catalog = ["http://example.com/image1.jpg", "http://example.com/image2.jpg", ...]
   sampler = DistributedRandomSampler(len(catalog))
   builder = build_data_pipeline(
       catalog,
       sampler,
       load_fn=load_data,
       preprocess_fn=process_data,
       collate_fn=collate_batch,
       batch_size=32,
       num_io_workers=16,
       num_cpu_workers=8,
   )

   # Create and use the pipeline
   pipeline = builder.build(num_threads=1)
   with pipeline.auto_stop():
       for batch in pipeline:
           # batch is a torch.Tensor on GPU, ready for training
           train_step(batch)

Building a DataLoader-Style Interface
--------------------------------------

Now that we've seen how to build a pipeline from a PyTorch Dataset, we can wrap this pattern in a reusable DataLoader class. This class follows the same structure as the ``build_data_pipeline`` function above, but provides a familiar PyTorch-style interface.

.. note::

   The DataLoader uses :py:class:`~spdl.source.DistributedRandomSampler` as the data source, which provides built-in support for distributed training. This sampler automatically handles data partitioning across multiple processes/nodes, ensuring each worker processes a unique subset of the data.

Here's a complete implementation:

.. code-block:: python

   from collections.abc import Iterator
   from typing import Callable, TypeVar
   from spdl.pipeline import PipelineBuilder
   from spdl.source import DistributedRandomSampler
   import spdl.io

   T = TypeVar('T')

   class DataLoader:
       """A PyTorch-style DataLoader built on SPDL Pipeline.

       This implementation follows the same staged approach as the
       build_data_pipeline function, with separate stages for I/O,
       CPU processing, batching, and GPU transfer.
       """

       def __init__(
           self,
           data_source: list,
           *,
           # Data processing functions
           load_fn: Callable[[int], T],
           preprocess_fn: Callable[[T], T] | None = None,
           collate_fn: Callable[[list[T]], torch.Tensor],
           # Batching
           batch_size: int,
           drop_last: bool = False,
           # Concurrency
           num_io_workers: int = 8,
           num_cpu_workers: int = 4,
           # Buffering
           buffer_size: int = 10,
       ):
           self.batch_size = batch_size
           self.drop_last = drop_last

           # Create sampler once for distributed training support
           self._sampler = DistributedRandomSampler(len(data_source))

           # Compute and cache the number of batches
           num_samples = len(self._sampler)
           if drop_last:
               self._num_batches = num_samples // batch_size
           else:
               self._num_batches = (num_samples + batch_size - 1) // batch_size

           # Build the pipeline builder using build_data_pipeline
           self._builder = build_data_pipeline(
               data_source,
               self._sampler,
               load_fn=load_fn,
               preprocess_fn=preprocess_fn,
               collate_fn=collate_fn,
               batch_size=batch_size,
               drop_last=drop_last,
               num_io_workers=num_io_workers,
               num_cpu_workers=num_cpu_workers,
               buffer_size=buffer_size,
           )

       def __len__(self) -> int:
           """Return the number of batches in the dataloader."""
           return self._num_batches

       def __iter__(self) -> Iterator:
           # Build a fresh pipeline for each iteration
           pipeline = self._builder.build(num_threads=1)
           with pipeline.auto_stop():
               yield from pipeline

Usage example:

.. code-block:: python

   # Define your processing functions
   def load_image(url: str) -> bytes:
       # I/O operation
       return download_from_url(url)

   def preprocess_image(data: bytes) -> np.ndarray:
       # CPU operations
       img = decode_image(data)
       img = resize(img, (224, 224))
       return img

   def collate_images(images: list[np.ndarray]) -> torch.Tensor:
       return torch.stack([torch.from_numpy(img) for img in images])

   # Create the dataloader
   dataloader = DataLoader(
       data_source=image_urls,
       load_fn=load_image,
       preprocess_fn=preprocess_image,
       collate_fn=collate_images,
       batch_size=32,
       num_io_workers=16,
       num_cpu_workers=8,
   )

   # Use it like PyTorch DataLoader
   for batch in dataloader:
       # batch is a torch.Tensor of shape (32, 224, 224, 3) on GPU
       train_step(batch)

Best Practices
--------------

1. **Profile Your Pipeline**

   SPDL provides powerful profiling tools to identify bottlenecks in your data loading pipeline. Use :py:func:`spdl.pipeline.profile_pipeline` to get detailed performance metrics for each stage:

   .. code-block:: python

      from spdl.pipeline import profile_pipeline

      # Profile the pipeline to see stage-by-stage performance
      profile_pipeline(
          builder.get_config(),
          num_iterations=100,  # Number of batches to profile
      )

   The :py:func:`~spdl.pipeline.profile_pipeline` function executes your pipeline for a specified number of iterations and reports detailed statistics including:

   - Throughput (items/second) for each stage
   - Time spent in each stage
   - Queue utilization between stages
   - Bottleneck identification

   For continuous monitoring during training, you can also enable runtime statistics:

   .. code-block:: python

      pipeline = builder.build(
          num_threads=1,
          report_stats_interval=5.0,  # Report every 5 seconds
      )

   For more details on performance analysis, see :doc:`../optimization_guide/analysis`.

2. **Tune Concurrency Levels**

   - Start with high concurrency for I/O operations (16-32 workers)
   - Use moderate concurrency for CPU operations (4-8 workers, matching CPU cores)
   - Keep CPU-to-GPU data transfer serial (no concurrency) because the underlying hardware does not support concurrent data transfer in one direction, and you want to avoid creating too many CUDA streams

3. **Choose Appropriate Output Order**

   - Use ``output_order="completion"`` for I/O stages to avoid head-of-line blocking
   - Use ``output_order="input"`` for preprocessing to maintain deterministic ordering

4. **Understanding Buffer Size**

   The ``buffer_size`` parameter in SPDL is somewhat analogous to PyTorch DataLoader's ``prefetch_factor``, but with important differences:

   **Key Differences:**

   - **Scaling behavior**: In PyTorch, the total number of prefetched batches is ``prefetch_factor × num_workers``. In SPDL, ``buffer_size`` is independent of the number of workers—it simply sets the sink buffer capacity.

   - **Performance impact**: In PyTorch, ``prefetch_factor`` directly affects data loading performance by controlling how many batches are prepared ahead of time. In SPDL, the pipeline continuously tries to fill all queues between stages, and the stage concurrency parameters act as the effective prefetch at each stage. The ``buffer_size`` does **not** affect pipeline performance—if the pipeline is fast enough, the buffer stays filled; if it's slow, the buffer remains mostly empty regardless of its size.

   **Recommendation**: Set ``buffer_size`` based on memory constraints rather than performance tuning. A value of 2-10 is typically sufficient. The actual prefetching happens at each pipeline stage based on the ``concurrency`` parameter.

5. **Consider Using Async Operations**

   For I/O-bound operations, async functions can be more efficient than synchronous alternatives. SPDL natively supports async functions—simply pass them directly to the pipeline:

   .. code-block:: python

      async def async_load(url: str) -> bytes:
          async with aiohttp.ClientSession() as session:
              async with session.get(url) as response:
                  return await response.read()

      builder.pipe(
          async_load,
          concurrency=32,  # Can handle many concurrent requests
      )

   .. important::

      **Do not wrap async functions in synchronous wrappers.** It's common to see async functions converted to sync by launching and shutting down an event loop at each invocation (e.g., using ``asyncio.run()``). This is inefficient and unnecessary in SPDL. The pipeline manages the event loop internally, so you should pass async functions as-is. SPDL will execute them efficiently using a shared event loop.

Comparison with PyTorch DataLoader
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - PyTorch DataLoader
     - SPDL Pipeline
   * - Processing Model
     - Monolithic Dataset
     - Staged Pipeline
   * - Concurrency
     - Process-based (multiprocessing)
     - Multi-threading, multiprocessing, async, and sub-interpreters (Python 3.14+)
   * - Initialization Overhead
     - High (dataset copied to each worker process)
     - Low
   * - Memory Overhead
     - High (each worker process holds a copy of the dataset)
     - Low
   * - Configurability
     - Limited (global num_workers)
     - Fine-grained (per-stage)
   * - I/O Optimization
     - Limited
     - Granular control with native asyncio support for high throughput
   * - Learning Curve
     - Familiar to PyTorch users
     - Requires understanding stages

When to Use SPDL
----------------

SPDL pipelines are particularly beneficial when:

- Your data loading involves significant I/O operations (network, remote storage)
- You need fine-grained control over different processing stages
- Memory efficiency is important (large batches, limited RAM)
- You want to optimize for throughput in production environments
- Your preprocessing involves mixed I/O and CPU operations

For simple datasets with minimal I/O and preprocessing, PyTorch's DataLoader may be sufficient. However, as your data loading becomes more complex, SPDL's staged approach provides better performance and flexibility.

Next Steps
----------

- See :doc:`../migration/pytorch` for detailed migration examples
- Explore :doc:`../case_studies/index` for real-world use cases
- Read :doc:`../optimization_guide/index` for performance tuning
