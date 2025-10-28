.. _noisy-neighbour:

Noisy Neighbour
===============

In the previous section, we used caching to estimate the upper bound of
performance gains achievable by optimizing the data loading pipeline.

It is intuitive that caching improves the performance of the training loop.
The batches are available in memory, eliminating wait time for batch creation.
The training loop only needs to look up the batch in memory.

However, there is another factor that enhances the performance of the
training loop, which is the lower CPU utilization.

When a training pipeline efficiently uses GPUs, the following occurs.

1. Whenever a unit of computation (GPU kernel) is completed,
   the CPU can launch the next one as soon as possible.
2. The data required to launch the next GPU kernel is available
   before the current GPU kernel is finished.

The goal of data loading optimization is to achieve the second part.
However, when CPU utilization is high, it becomes difficult for the
CPU to launch GPU kernels in a timely manner.
We refer to this phenomenon as the "noisy neighbour".

To show the effect of the noisy neighbour, we conducted an experiment.
We ran a pipeline that trains an image classification model.
After some steps, we spawned a subprocess that serves no functionality but consumes
CPU resource.
We repeat this until the CPU utilization hit 100%.

The following plot shows how the training speed (batches per second) drops
as we add increasing CPU loads.

.. raw:: html

   <div id='nn_exp'></div>


.. include:: ../plots/noisy_neighbour.txt

This suggests that data loading needs to be not only fast but also efficient.
At one point, we thought that since we are using GPUs for model computation,
we could use CPU resources for data loading and should utilize
the CPU as much as possible.
This turned out to be an anti-pattern.

We now recommend keeping CPU utilization at a maximum of 40% for efficient
training.
