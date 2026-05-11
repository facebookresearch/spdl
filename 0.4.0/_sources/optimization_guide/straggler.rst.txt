The Straggler
=============

.. image:: ../../_static/data/straggler_illustration.png

In distributed training, there are cases where
one rank is significantly slower than the rest,
which degrades the overall training speed.


Such a rank is sometimes called a "straggler."
When there is a straggler, we often observe that
``nccl:all_reduce`` takes a significant amount of time on other ranks.

The following figure shows an example of
one rank waiting for ``nccl:all_reduce`` to complete.
When this happens, the straggler causes other ranks to wait for it to
catch up on loss computation.

.. image:: ../../_static/data/straggler_trace_others.png

There are many reasons why one rank can be slower than the rest.
For example, hardware failure is a common issue in large-scale distributed training.

Slow data loading can also cause a straggler.

The following figure shows the trace of the straggler from the same trace
as the previous one.
It is blocked on ``next(dataloader)``.

.. image:: ../../_static/data/straggler_trace.png

When slow data loading is the cause,
we often observe that the rank of the straggler changes from epoch to epoch.

The following is a plot of :ref:`data-readiness`
for a distributed training setup consisting of 8 ranks.

.. raw:: html

   <div id="straggler_data_readiness"></div>

.. include:: ../plots/straggler.txt

The data readiness at the sink stage indicates whether
data loading is faster than the training process.
A value of 1 means data loading is fast enough,
and 0 means data loading is the bottleneck.

The plot suggests that there is one rank that is not fast enough (a straggler),
but the struggling rank changes from epoch to epoch.
Earlier in the training, rank 4 is the slowest, but later, rank 0 becomes the slowest.

How does this happen?
It is quite simple.

When the pipeline starts, one rank can be slightly slower than the others.
The other ranks gain a window of time while waiting for the slower rank to
do additional data loading work, putting them ahead of the slow one.

As training progresses, the difference between the slow rank and the others
becomes significant, and the slow rank becomes the straggler.

Since this difference arises from subtle perturbations in computation speed,
any rank can become the straggler.
However, rank 0 is more susceptible than the others because
it usually has more work than the rest.

When a straggler is found and the straggler changes each epoch,
it indicates that data loading is not fast enough.
