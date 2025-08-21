IO Module
=========

The ``spdl.io`` module is a stand-alone module implements efficiently loading data into array formats.

It is mainly designed for AI training/inference running in a cloud.
It supports the convert various data format in byte string (downloaded from a remote storage)
into array format, and transfer it to GPU without interrupting the model computation in GPU.
The following diagram illustrates this.

.. mermaid::

   flowchart TD
      subgraph C[Compute Node]
        b(Binary String 
        &#40video, audio, image, NumPy etc...&#41) e1@<==> ct(CPU array)
        subgraph G[GPU]
            gt(GPU array)
        end
	ct e2@<==> gt

        e1@{ animate: true }
        e2@{ animate: true }
      end
      A[Remote Storage] --> |Data| C


The IO module contains efficient implementations for commonly used formats.
We will see in detail how they are implemented.

.. toctree::
   :maxdepth: 3

   multi_media
   numpy
   transfer
