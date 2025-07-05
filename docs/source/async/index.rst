Introduction to Async I/O
=========================

Asynchronous operations (a.k.a. async I/O) are the foundation of SPDL,
but many ML practitioners are not familiar with such programming paradigm.

For advancced usage of SPDL, it is important to understand how async IO works,
because inproper usage of async I/O can slow down the whole system.

In this section, we share our view of what problem async I/O solves and how
it evolved from concurrent programming in synchronous domain, in the hope
it helps you understand how async I/O (thus the internal of SPDL pipeline)
works and it lets you take advantage of it.


.. toctree::

   problem
