IO Module
=========

The ``spdl.io`` module is a stand-alone module that implements efficient loading of data into array formats.

It is mainly designed for AI training/inference running in the cloud.
It supports converting various data formats in byte strings (downloaded from remote storage)
into array format, and transferring it to GPU without interrupting the model computation on GPU.

The following diagram illustrates this workflow:

.. mermaid::

   flowchart
      subgraph C[Compute Node]
        direction LR
        b(Binary String&#10&#40video, audio, image, NumPy etc...&#41)
        c(CPU array)

        subgraph G[GPU]
            g(GPU array)
        end

        b e1@<==> |<b>Decoding</b>| c
        c e2@<==> |<b>Transfer</b>| g

        e1@{ animation: slow }
        e2@{ animation: slow }
      end


The IO module contains efficient implementations for commonly used formats.
This section provides detailed documentation on how to use these implementations.

.. toctree::
   :maxdepth: 2

   basic
