Memory and transport
====================

Shared-memory arenas let processes reuse transport storage. Choose the workflow
based on where the bytes are produced and who owns the storage:

- For a subprocess iterable, compare pool-backed views with ring-buffer
  copy-out. The shared-memory arenas guide covers sizing, lifetime, and
  optional pinning for GPU transfer.
- For a process region in a pipeline, pass caller-owned arenas at the input or
  output boundary to change transport without rewriting the stage.
- If a stage can write into an output buffer, direct allocation can also remove
  the copy from a private result into the arena. It has a separate reservation
  and ticket lifecycle.

.. toctree::
   :maxdepth: 1

   shared_memory_arenas
   transparent_region_arenas
   direct_allocations
