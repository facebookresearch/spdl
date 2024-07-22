Which stage is the bottleneck?
==============================

.. py:currentmodule:: spdl.dataloader

By default, when a pipe stage is created, :py:class:`TaskStatsHook` is attached to the stage.
This hook collects runtime statistics of the stage and report them at the end.
We can use this report to determine which stage in the pipeline is being the bottleneck
thus requires improvement.

Task Stats and QPS
------------------

Let's say, we have a pipeline that downloads data from remote storage system and process the data.

.. code-block::

   >>> pipeline = (
   ...     PipelineBuilder()
   ...     .add_source(sample_list)
   ...     .pipe(download, concurrency=32)
   ...     .pipe(process, concurrency=128)
   ...     .add_sink(3)
   ...     .build(num_threads=128)
   ... )

We execute the pipeline without downstream load, and we obtain the following log.

.. code-block::

   >>> with pipeline.auto_stop():
   ...     for data in pipeline:
   ...         pass

.. code-block::

   [download]	Completed  2559 tasks (163 failed) in 17.3974 [sec]. QPS: 137.72 (Concurrency:  32). Average task time: 128.7163 [ ms].
   [process]	Completed  2396 tasks (  0 failed) in 18.9318 [sec]. QPS: 126.56 (Concurrency: 128). Average task time: 896.3629 [ ms].

:py:class:`TaskStatsHook` collects the number of tasks (items processed by the stage), the average task execution time and the time elapsed between the start and the end of the stage.

.. note::

   The average task time does not include the failed tasks, however,
   the total duration of the stage include everything from task execution
   to the time it waited on input/output queues.

``QPS`` (queries per second, though it is not query) is the number of tasks successfully completed, devided by the duration of the stage. It is an indication of how fast the stage is processing items.

From source to the sink, ``QPS`` can only decrease, as it is impossible for downstream stages to process items faster than upstream stages.
If a downstream stage is fast enough to catch up its upstream stage, then the QPS values are raoughly the same between these stages.
If a downstream stage is not as fast as its upstream stage, then the QPS value of the downstream stage is smaller than that of the upstream stage.

Therefore, by locating the stage at which QPS drops significantly, we can determine the stage which is the bottleneck within the pipline.

In the above example, QPS drops from 137 to 126 in the ``process`` stage. This indicates that data processing is not up to the speed of data acquisition. To optimize the pipeline throughput, one needs to improve the performance of the ``process`` stage. Increasing the throughput of the ``download`` stage does not help. How to optimize the ``process`` stage depends on the other factors. For example if the machine executing the pipeline has spare computation resource, then increasing the concurrency and the number of threads can help.

Tuning Pipeline Performance
---------------------------

Let's look at another example.

This time, we download images from the remote source, decode, batch and send them to GPU device. We create decoding/preprocessing and batch stage using :py:mod:`spdl.io`.

.. code-block::

   >>> def decode(width=224, height=224, pix_fmt="rgb24") -> Callable:
   ...     """Decode and resize image from byte string"""
   ...     filter_desc = spdl.io.get_video_filter_desc(
   ...         scale_width=width, scale_height=height, pix_fmt=pix_fmt
   ...     )
   ...
   ...     async def decode(data: bytes) -> FFmpegFrames:
   ...         packets = await spdl.io.async_demux_image(data)
   ...         frames = await spdl.io.async_decode_packets(packets, filter_desc=filter_desc)
   ...         return frames
   ...
   ...     return decode

.. code-block::

   >>> async def batchify(frames: list[FFmpegFrames]) -> torch.Tensor:
   ...     """Create a batch from image frames, send them to GPU and create Torch tensor"""
   ...     cfg = spdl.io.cuda_config(device_index=0)
   ...     cpu_buffer = await spdl.io.async_convert_frames(frames)
   ...     cuda_buffer = await spdl.io.async_transfer_buffer(cpu_buffer, cuda_config=cfg)
   ...
   ...     return routes, spdl.io.to_torch(cuda_buffer)

.. code-block::

   >>> def run_pipeline(dl_concurrency, decode_concurrency):
   ...     pipeline = (
   ...         PipelineBuilder()
   ...         .add_source(src)
   ...         .pipe(download, concurrency=dl_concurrency)
   ...         .pipe(decode(), concurrency=decode_concurrency)
   ...         .aggregate(32)
   ...         .pipe(batchify)
   ...         .add_sink(10)
   ...         .build(num_threads=decode_concurrency)
   ...     )
   ...     with pipeline.auto_stop():
   ...         for item in pipeline:
   ...             pass


Now we run the pipeline with different concurrency values for downloading and decoding.
(You can skip the raw result and go to the summary table bellow.)

.. code-block::

   >>> run_pipeline(64, 4)

.. code-block::

   [download]	Completed  1600 tasks (  0 failed) in 6.2723 [sec]. QPS: 255.09 (Concurrency:  64). Average task time: 240.2614 [ ms].
   [decode_image]	Completed  1600 tasks (  0 failed) in 6.2763 [sec]. QPS: 254.93 (Concurrency:   4). Average task time: 3.8516 [ ms].
   [aggregate(32, drop_last=False)]	Completed  1601 tasks (  0 failed) in 6.2786 [sec]. QPS: 254.99 (Concurrency:   1). Average task time: 0.0038 [ ms].
   [batchify]	Completed    50 tasks (  0 failed) in 6.2790 [sec]. QPS: 7.96 (Concurrency:   1). Average task time: 1.6127 [ ms].
   [sink]	Processed    50 items in 6.2799 [sec]. QPS: 7.96. Average wait time: Upstream: 123.1203 [ms ], Downstream: 0.0043 [ms ].

.. code-block::

   >>> run_pipeline(128, 4)

.. code-block::

   [download]	Completed  1600 tasks (  0 failed) in 2.0347 [sec]. QPS: 786.36 (Concurrency: 128). Average task time: 139.2847 [ ms].
   [decode_image]	Completed  1600 tasks (  0 failed) in 2.0385 [sec]. QPS: 784.89 (Concurrency:   4). Average task time: 4.1260 [ ms].
   [aggregate(32, drop_last=False)]	Completed  1601 tasks (  0 failed) in 2.0477 [sec]. QPS: 781.87 (Concurrency:   1). Average task time: 0.0039 [ ms].
   [batchify]	Completed    50 tasks (  0 failed) in 2.0522 [sec]. QPS: 24.36 (Concurrency:   1). Average task time: 2.6672 [ ms].
   [sink]	Processed    50 items in 2.0529 [sec]. QPS: 24.36. Average wait time: Upstream: 40.2405 [ms ], Downstream: 0.0040 [ms ].

.. code-block::

   >>> run_pipeline(256, 4)

.. code-block::

   [download]	Completed  1600 tasks (  0 failed) in 1.8855 [sec]. QPS: 848.57 (Concurrency: 256). Average task time: 146.4174 [ ms].
   [decode_image]	Completed  1600 tasks (  0 failed) in 1.8907 [sec]. QPS: 846.25 (Concurrency:   4). Average task time: 4.3023 [ ms].
   [aggregate(32, drop_last=False)]	Completed  1601 tasks (  0 failed) in 1.8935 [sec]. QPS: 845.52 (Concurrency:   1). Average task time: 0.0038 [ ms].
   [batchify]	Completed    50 tasks (  0 failed) in 1.8942 [sec]. QPS: 26.40 (Concurrency:   1). Average task time: 2.6016 [ ms].
   [sink]	Processed    50 items in 1.8945 [sec]. QPS: 26.39. Average wait time: Upstream: 37.1349 [ms ], Downstream: 0.0039 [ms ].

.. code-block::

   >>> run_pipeline(512, 4)

.. code-block::

   [download]	Completed  1600 tasks (  0 failed) in 1.9942 [sec]. QPS: 802.31 (Concurrency: 512). Average task time: 151.8697 [ ms].
   [decode_image]	Completed  1600 tasks (  0 failed) in 1.9986 [sec]. QPS: 800.55 (Concurrency:   4). Average task time: 4.5500 [ ms].
   [aggregate(32, drop_last=False)]	Completed  1601 tasks (  0 failed) in 2.0021 [sec]. QPS: 799.67 (Concurrency:   1). Average task time: 0.0038 [ ms].
   [batchify]	Completed    50 tasks (  0 failed) in 2.0029 [sec]. QPS: 24.96 (Concurrency:   1). Average task time: 3.1626 [ ms].
   [sink]	Processed    50 items in 2.0037 [sec]. QPS: 24.95. Average wait time: Upstream: 39.2747 [ms ], Downstream: 0.0044 [ms ].

.. code-block::

   >>> run_pipeline(256, 8)

.. code-block::

   [download]	Completed  1600 tasks (  0 failed) in 1.3731 [sec]. QPS: 1165.27 (Concurrency: 256). Average task time: 152.5442 [ ms].
   [decode_image]	Completed  1600 tasks (  0 failed) in 1.3768 [sec]. QPS: 1162.10 (Concurrency:   8). Average task time: 5.4827 [ ms].
   [aggregate(32, drop_last=False)]	Completed  1601 tasks (  0 failed) in 1.3794 [sec]. QPS: 1160.61 (Concurrency:   1). Average task time: 0.0038 [ ms].
   [batchify]	Completed    50 tasks (  0 failed) in 1.3806 [sec]. QPS: 36.22 (Concurrency:   1). Average task time: 3.1528 [ ms].
   [sink]	Processed    50 items in 1.3814 [sec]. QPS: 36.20. Average wait time: Upstream: 27.0740 [ms ], Downstream: 0.0041 [ms ].

.. code-block::

   >>> run_pipeline(256, 16)

.. code-block::

   [download]	Completed  1600 tasks (  0 failed) in 2.2429 [sec]. QPS: 713.36 (Concurrency: 256). Average task time: 154.0344 [ ms].
   [decode_image]	Completed  1600 tasks (  0 failed) in 2.2661 [sec]. QPS: 706.06 (Concurrency:  16). Average task time: 14.4060 [ ms].
   [aggregate(32, drop_last=False)]	Completed  1601 tasks (  0 failed) in 2.3514 [sec]. QPS: 680.86 (Concurrency:   1). Average task time: 0.0039 [ ms].
   [batchify]	Completed    50 tasks (  0 failed) in 2.3610 [sec]. QPS: 21.18 (Concurrency:   1). Average task time: 15.5912 [ ms].
   [sink]	Processed    50 items in 2.3622 [sec]. QPS: 21.17. Average wait time: Upstream: 46.3030 [ms ], Downstream: 0.0041 [ms ].

.. code-block::

   >>> run_pipeline(256, 32)

.. code-block::

   [download]	Completed  1600 tasks (  0 failed) in 1.6766 [sec]. QPS: 954.30 (Concurrency: 256). Average task time: 156.4433 [ ms].
   [decode_image]	Completed  1600 tasks (  0 failed) in 1.6815 [sec]. QPS: 951.55 (Concurrency:  32). Average task time: 18.5734 [ ms].
   [aggregate(32, drop_last=False)]	Completed  1601 tasks (  0 failed) in 1.6862 [sec]. QPS: 949.48 (Concurrency:   1). Average task time: 0.0039 [ ms].
   [batchify]	Completed    50 tasks (  0 failed) in 1.6866 [sec]. QPS: 29.64 (Concurrency:   1). Average task time: 15.2163 [ ms].
   [sink]	Processed    50 items in 1.6913 [sec]. QPS: 29.56. Average wait time: Upstream: 33.1498 [ms ], Downstream: 0.0041 [ms ].

The following table summarizes the above result.

.. table::
   :class: right-align

   === ==================== ================== ============ ============ =====================
   Run Download Concurrency Decode Concurrency Download QPS Decoding QPS Sink QPS (normalized)
   === ==================== ================== ============ ============ =====================
     1                   64                  4       255.09       254.93                254.72
     2                  128                  4       786.36       784.89                779.52
     3                  256                  4       848.57       846.25                844.48
     4                  512                  4       802.31       800.55                798.40
     5                  256                  8      1165.27      1162.10               1158.40
     6                  256                 16       713.36       706.06                677.44
     7                  256                 32       954.30       951.55                945.92
   === ==================== ================== ============ ============ =====================

Looking at the first pipeline (``Run 1``), we do not see a siginficant QPS drop in the stages.
It is around 241 at the beginning and the at the end of the pipeline.
This suggests that the firsta stage (download) is dominating the QPS of the whole pipeline.
So we increase the download concurrency.

As we increase the concurrency of download (``Run 2 - 4``), QPS increases, but QPS is saturated
around 800.
Because the pipeline is automatically blocked according to the performance of the downstream,
we tweak the concurrency of decoding.
Increasing the decode concurrency from 4 to 8 (``Run 5``), the QPS increases further more, but
it drops again beyond 16 (``Run 6, 7``).

Summary
-------

When running :py:class:`Pipeline`, :py:class:`TaskStatsHook`: provides runtime statistics of stages. This information is helpful when determining which part of the pipeline should be optimized.
