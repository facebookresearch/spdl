Remote Iterable Protocol
========================

.. py:currentmodule:: spdl.pipeline._iter_utils

Manipulting an iterable object in a remote location (subprocess or subinterpreter) requires
somewhat elaborated state control.
The following section go over the implementation detail.

Worker State
------------

The iterable object is manipulated in the worker process.
The worker process has three states, ``Initialization``, ``Stand By`` and ``Iteration``.

The Initialization state performs global initialization and create the iterable object.
When the Initialization completes, the worker transition to ``Stand By`` mode,
where it waits for a command from the parent process.

The command can be ``START_ITERATION`` or ``ABORT``.
When the ``START_ITERATION`` is received, the worker process transition to the Iteration mode.

In the Iteration mode, the worker creates an iterator object from the iterable, then executes it.
The resulting data are put in the queue, which the parent process is watching.

The following diagram illustrates worker's state transition in simplified manner.
Detailed diagram alongside the actual implementation is found in :py:func:`_execute_iterable`.

.. mermaid::

    stateDiagram-v2
        state Parent {
            p1: Start Iteration
            p2: Iterate on the result
            state pf <<fork>>
            state pj <<join>>

            [*] --> p1
            p1 --> pf
            pf --> pj: Wait for worker process
            pj -->  p2
            p2 --> [*]
        }

        state Worker {
            state wf <<fork>>
            w0: Initialization
            w1: Stand By
            w2: Iteration

            [*]--> w0
            w0 --> w1: Success
            w0 --> [*]: Fail
            w1 --> wf: Iteration started
            wf --> w2
            w2 --> w1: Iteration completed

            w1 --> [*]: Abort
            w2 --> [*]: Fail / Abort
        }
        pf --> w1: Issue START_ITERATION command
        wf --> pj: Notify ITERATION_STARTED
        w2 --> p2: Results passed via queue

Helper functions and data structures
-------------------------------------

The follosing functions and data structures are used to implement
the :py:func:`~spdl.pipeline.iterate_in_subprocess` and :py:func:`iterate_in_subinterpreter` functions.
They are not public interface, but the logic is sufficiently elaborated,
and it is helpful to have them in the documentation, so they are listed here.

.. autoclass:: _Cmd
   :noindex:
   :members:

.. autoclass:: _Status
   :noindex:
   :members:

.. autofunction:: _enter_iteration_mode()
   :noindex:

.. autofunction:: _execute_iterable()
   :noindex:

.. autofunction:: _drain()
   :noindex:

.. autofunction:: _iterate_results()
   :noindex:

.. autoclass:: _SubprocessIterable()
   :noindex:
   :members: __iter__
