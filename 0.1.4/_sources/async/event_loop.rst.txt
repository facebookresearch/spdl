The Event Loop
==============

In the previous section, we attempted to build an asynchronous orchestration
mechanism using a thread pool executor, and it became fairly complex.

In this section, we look at how the event loop, the core abstraction of the async I/O,
can make it simple.
As a result, it allows to write complex asynchronous procedure in a form
close to the usual synchronous procedual programming.

Now let's look at the code we had in the previous section again.

.. code-block::

   futures = {executor.submit(task...) for task in tasks}

   while futures:
       done, futures = wait(tasks, return_when=FIRST_COMPLETED)

       for future in done:
           try:
               result = future.result()
           except Exception:
               # A task failed
               ...
           else:
               # A task succeed
               # Check and fetch the next stage
               if (fn := get_next_task(result)) is not None:
                   # Invoke
                   future = executor.submit(fn, result)
                   futures.add(future)


This code is responsible for orchestrating concurrent task execution, so it is
agnostic to tasks themselves.
The orchestration must be efficient so that it won't add performance overhead.
It is often said "don't reinvent the wheel", but people realized that
building an abstraction that efficiently handles orchestration is worthwhile.

The resulting abstraction (and a concrete implementation) is called the event
loop, and it is the core abstraction of async I/O.

The event loop is a mechanism which **schedules new tasks** and
**reacts when a task completes**.
From the user perspective, instead of actively attending to Future objects,
the event loop gets notified when a task is completed.

The following diagram is our attempt at depicting the behavior of the event loop.

.. image:: ../_static/data/event_loop_1.png
   :width: 500

The main branch of the program execution does not attend to an on-going task.
Instead, it waits for a task to be completed.
When a task is completed, the event loop schedules the callback,
which is another task, then goes back to the stand-by state.
It repeats the process until there are no more tasks.

Python's :py:mod:`asyncio` module implements the event loop using low-level
primitives provided by the OS†.
So we can expect it to be very efficient.

.. admonition:: † The underlying multiplexing mechanism.
   :class: note

   You can find the default mechanism with :py:class:`selectors.DefaultSelector`.
   On Linux, this points :py:class:`selectors.EpollSelector` and on macOS,
   this points :py:class:`selectors.KqueueSelector`.

Using `PyTorch Profiler <https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html>`_, we can look at what event loop does.

The following screenshot is the trace of :py:func:`asyncio.run`, which runs
the event loop.
The repeated pink stripes (annotated with blue circles) are where
the event loop is waiting for a task to complete.
The occasional patterns that look like icicles (annorated with red rectangulars)
are where new tasks are scheduled.

.. image:: ../_static/data/event_loop_2.png

If we zoom-in, we can see what task is being scheduled.
           
.. image:: ../_static/data/event_loop_3.png

In the above case, it is a function passed to the
:py:meth:`spdl.pipeline.PipelineBuilder.pipe` method.
We can see that the scheduling takes about 340 nanoseconds.
In this pipeline, a new event happens about every 50~100 milliseconds,
so the overhead of task scheduling should be negligible.

We will look into it in the next section, but it is important to know that
the event loop can schedule only one task at a time.
