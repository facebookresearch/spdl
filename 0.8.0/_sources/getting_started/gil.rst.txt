.. _working-around-the-gil:

Working Around the GIL
======================

In Python, the GIL (Global Interpreter Lock) practically prevents multi-threaded
code from running Python bytecode in parallel: while one thread holds the lock,
no other thread in the same process can execute Python. Extension modules written
in low-level languages such as C, C++ and Rust can, however, release the GIL
while executing operations that do not interact with the Python interpreter.

This page collects GIL-specific guidance. For the execution models that build on
it, see :ref:`pipeline-parallelism` (the fundamentals) and :ref:`execution-models`
(the MT / MP / MTP patterns).

Which operations release the GIL?
---------------------------------

Many libraries used for data loading release the GIL. To name a few:

- Pillow
- OpenCV
- Decord
- tiktoken
- polars

Libraries such as PyTorch and NumPy also release the GIL when manipulating
arrays, so they are usually fine. For loading raw byte strings into array format,
SPDL offers efficient GIL-releasing functions through the :py:mod:`spdl.io`
module.

Typically the bottleneck in model training is loading and pre-processing media
data. Even though some parts of a pipeline are constrained by the GIL, we can
achieve high throughput by using pre-processing functions that release it. This
is what lets the default multi-threaded pipeline scale; see
:ref:`pipeline-parallelism` for how the shared thread pool dispatches these
stages.

Example: pandas vs polars
-------------------------

DataFrame libraries make the effect concrete. `polars <https://pola.rs/>`_ is
implemented in Rust and releases the GIL during its operations, whereas
`pandas <https://pandas.pydata.org/>`_ is largely Python/Cython and holds the GIL
for much of its work.

In a multi-threaded pipeline (MT or MTP) whose per-row work is a DataFrame decode
and transform, switching the backend from pandas to polars can nearly double
end-to-end throughput -- we observed roughly a 1.8x speedup on such a workload --
because the polars stages run in parallel across the thread pool while the pandas
stages are serialized by the GIL.

.. figure:: ../_static/data/pandas_vs_polars.png
   :width: 100%

   The same pipeline with a pandas vs a polars backend (higher is better;
   absolute values omitted, gridlines for scale). On the GIL builds (3.12, 3.14)
   polars roughly doubles the threaded runners (mt, mtp), while multi-processing
   (mp) is largely unchanged. On free-threaded Python (3.14t; no polars wheel yet)
   pandas alone reaches comparable threaded throughput, since the GIL no longer
   serializes the threads.

The gain is specific to the thread-based models. With multi-processing (MP),
where each worker has its own interpreter and GIL, the backend choice makes
little difference: the parallelism there does not depend on the stage releasing
the GIL. This is a good illustration of why the best execution model depends on
whether the hot stage releases the GIL; see :ref:`execution-models`.

What if a function does not release the GIL?
--------------------------------------------

If a stage relies on a function that holds the GIL, running it on the shared
thread pool blocks the other stages. In that case, move the work off the shared
thread pool:

- Delegate the single stage to a subprocess with a
  :py:class:`~concurrent.futures.ProcessPoolExecutor`. See the "Multi-processing
  (stage)" section of :ref:`pipeline-parallelism` for the mechanics, including the
  picklability requirements and one-time subprocess initialization.
- For a data loader with several such stages, use the multi-processing (MP) or
  multi-threading-in-subprocess (MTP) patterns instead of paying a process
  round trip per stage. See :ref:`execution-models`.

On free-threaded (no-GIL) builds of Python the constraint is lifted, so threaded
stages can run Python in parallel; see :ref:`execution-models` for how that
affects the choice of execution model.

Which functions hold the GIL?
-----------------------------

The following is the list of functions that we are aware hold the GIL. It is
advised to use them with a ``ProcessPoolExecutor`` (as above) or to avoid using
them in SPDL.

* `np.load <https://github.com/numpy/numpy/blob/maintenance/2.1.x/numpy/lib/_npyio_impl.py#L312-L500>`_: Please refer to :ref:`data-formats-case-study` for a possible workaround.
