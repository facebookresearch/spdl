.. _working-with-agents:

Working with Agents
===================

SPDL's optimization workflow is a loop: run the pipeline, read the per-stage
runtime statistics, find the stage that limits the throughput, change it,
and validate the result.
The loop was designed for engineers, and running it well used to require
deep knowledge of data loading.
That knowledge is now written down in the
:ref:`Knowledge Base <optimization-knowledge-base>`, so a coding agent can
run the same loop.

This page describes how to set up such a workflow.

Why the loop suits an agent
---------------------------

The properties of SPDL that help engineers optimize a pipeline also help
agents.

- *The bottleneck shows up in numbers.*
  The per-stage statistics, such as how full the output queue of each stage
  is, point to the slow stage.
  An agent can choose its next experiment based on measurements rather than
  guesses.
  (See :ref:`analysis`.)
- *Changes stay small.*
  Each stage has its own concurrency and executor, so a change can be tested
  one at a time and its effect can be attributed.
- *The knowledge is written down.*
  The Knowledge Base covers migration, analysis and optimization techniques.
  Its :doc:`Technique catalog <knowledge_base/_generated/optimization/_categories/technique-catalog>`
  indexes each technique by the symptom that makes it worth testing, and
  lists the correctness checks and the resource costs to verify, including
  approaches that did not help.
  Its source is a set of Markdown files in ``tools/skills/`` of the SPDL
  repository, which coding agents can consume as-is.

What to give the agent
----------------------

- **Goal:** the metric to improve, such as samples per second per GPU or
  end-to-end training time.
- **Job:** the training job and the pipeline configuration.
- **Baseline:** a measurement of the current implementation under the same
  conditions as the planned experiments.
- **Checks:** the correctness requirements the change must pass, such as
  the same samples, transforms and batch contents, and the model quality.
- **Budget:** how much compute and time the agent can spend on experiments.

What the engineer does
----------------------

The agent does the hands-on work: writing code, running experiments and
analyzing the results.
The engineer sets the direction, steers at decision points, and reviews the
changes and the evidence behind them.

Ask the agent to explain which observations support each proposed change,
and to keep the results of unsuccessful attempts.
A negative result tells the next experiment what not to repeat.

A typical workflow
------------------

1. **Migrate.**
   Move the existing data loader to SPDL with the closest architecture.
   The :doc:`Migration guidance <knowledge_base/_generated/migration/_categories/migration-guidance>`
   reuses the functions and classes of the original implementation, and
   checks the data, resource usage and throughput against the original
   before any optimization.
   The :doc:`Startup audit <knowledge_base/_generated/_categories/startup-audit>`
   checks the recurring fetch calls for hidden setup costs.
2. **Analyze.**
   Run the pipeline and read the per-stage statistics to find the exposed
   cost. If the data loading is no longer the bottleneck, look at the model
   with traces.
   The :doc:`Optimization guidance <knowledge_base/_generated/optimization/_categories/optimization-guidance>`
   explains how to interpret the statistics and compare candidate changes.
3. **Improve.**
   Choose a technique from the
   :doc:`Technique catalog <knowledge_base/_generated/optimization/_categories/technique-catalog>`
   whose trigger matches the measurement, apply it, and validate the speed
   and the quality against the baseline.
   :doc:`Techniques by area <knowledge_base/_generated/optimization/techniques/_categories/techniques-by-area>`
   groups the same techniques by the part of the input path they change,
   alongside negative results.
   For the model side, the
   :doc:`Model optimization catalog <knowledge_base/_generated/optimization/model_techniques/_categories/model-optimization>`
   groups the reusable mechanisms by the measured profile, such as compute,
   memory or distributed waiting.
   Record each attempt with the technique's ID, whether it helped or not.
4. **Repeat.**
   Removing one bottleneck can expose another. Measure again after each
   change. Tune the parameters after the architecture is finalized.

Contributing back
-----------------

When an agent finds a pattern that helps, or a plausible approach that does
not, record the measured bottleneck, the mechanism, the comparison and the
correctness checks, and add it to the corresponding Technique catalog
entry, or propose a new entry.
Each case makes the guidance better for the next pipeline.
