.. _intro-async:

Introduction to Async I/O
=========================

Asynchronous operations are the foundation of SPDL,
and SPDL uses :py:mod:`asyncio` for orchestrating them.
For advancced usage of SPDL, it is important to understand
how the Async I/O works, because improper usage of
async I/O can slow down the whole system.
However, the async I/O is a programming paradigm not all Python
developers are familiar with.

This section is an introductory course of the Async I/O.
Instead of starting from the syntax, we start by illustrating issues
we encounter when we try to build an orchestration system
with the traditional multi-threading.

We look at how such issues are resolved by the Async I/O's event loop,
then introduce ``async def`` and ``await`` keywords.
Finally, we look at how synchronous functions can be
converted to Awaitable objects so that :py:mod:`asyncio` can handle
sync/async functions in unified manner.

.. toctree::

   problem
   event_loop
   unit
   sync
