SPDL in Inference pipeline
==========================

In this section, we explain how to run inference pipeline with SPDL.

The primal goal of the SPDL project is to solve the data loading bottleneck in model training.
We designed SPDL's Pipeline to be flexible and generic to support various data flow patterns.
It turned out that this generic design can naturally support inference tasks.

Inference pipelines involve loading data onto GPU, running inference,
moving the result from GPU to CPU, and saving the result.

The difference between training and inference is as follow.

- In the training, the model computation involves forward/backward path,
  and parameter update, and they are carried out in the foreground (main) thread.
- In the training, the model computation only performs the forward path.
  When running the inference with SPDL, the model computation
  (and data transfer between CPU and GPU) is performed in background threads.

Let's look into how to

.. raw:: html

   <div id="baseline_smu"></div>

.. raw:: html

   <div id="spdl_smu"></div>

.. include:: ../plots/inference.txt

