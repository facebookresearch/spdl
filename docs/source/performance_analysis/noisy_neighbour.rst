Noisy Neighbour
===============

In the previous section, we used caching to estimate the upper bound of performance
gain we can get by optimizing the data loading pipeline.

It is intuitive that caching improves the performance of training loop. The batches
are available in memory, so there is no wait time for the batch creation.
The training loop only needs to look up the batch in memory.

However, there is another factor that helps the performance of training loop.
That is the lower CPU utilization.

When a training pipeline is efficiently using GPUs, the following happens.

1. Whenever a unit of computate operation (GPU kernel) is completed, CPU is able to launch the next one as soon as possible.
2. The data required to launch the next GPU kernel is available before the current GPU kernel is finished.

The goal of data loading optimization is to achive the second part.
It turns out, however, when the CPU utilization is high, it becomes difficult
for the CPU to launch GPU kernels in timely manner.
We call this phenomenon "noisy neighbour".

To show the effect of the noisy neighbour, we conducted an experiment.
We ran a pipeline that trains an image classification model.
After some steps, we spawned a subprocess that serves no functionality but consumes
CPU resource.
We repeat this until the CPU utilization hit 100%.

The following plot shows the how the training speed (batch per second) drops as
we add more and more CPU loads.

.. raw:: html

   <div id='nn_exp'></div>


.. include:: ../plots/noisy_neighbour.txt

This suggests that data loading needs to be not only fast but also efficient.
At some point, we thought that since we are using GPUs for model computation, we
can use CPU resources for data loading, and we should be utilizing CPU as much as
possible. This turned out to be an anti-pattern.

We now recommend to keep the CPU utilization at most 40% for efficient training.
