Why migrate to SPDL?
====================

When migrating your data loading solution to SPDL, you should ask why you want to do that,
and what's the benefit and what's the drawback?

Here are the pros and cons of migrating to SPDL.

Pros
----

1. *Performance*
    When switching to SPDL, it is not uncommon to see x3 throughput in data loading.
    The improvement is not directly reflected to model training performance,
    but considering how uniquietous the data loading bottleneck is,
    there is a good chance adopting SPDL increases the model training performance.
2. *Efficiency*
    Switching to SPDL essentially means to switching from sub-process-based parallelism to
    thread-based parallelism.
    Thread-based parallelism uses smaller compute resource to achieve the same throughput
    as sub process-based parallelism.
    This leaves spare capacity for increased data loading demand in the future.
3. *Tunability*
    The :py:class:`spdl.dataloader.Pipeline` allows to configure the concurrency stage by stage.
    This makes the pipeline flexible to fit different environments that the pipeline is running.
    You can adjust them based on the network bandwidth and CPU capacity independently.
4. *Flexibility*
    The :py:class:`spdl.dataloader.Pipeline` executes the functions you provide. SPDL
    does not put any restriction on what data can go through the pipeline. Stages can
    aggregate/disaggregate data along the way.
5. *Observability*
    As we have seen in :ref:`Performance Analysis<Performance Analysis>` section, SPDL's
    pipeline abstraction gives insights of stage-wise runtime performance, which makes it
    easier to understand how the data loading is performing and how to optimize the pipeline.

Cons
----

1. *Onboarding Cost*
    Although we are working to make the onboarding easier, since SPDL involves paradigm shift
    (from sub-process-based parallelism to thread-based parallelism, and from object-oriented
    composition to functional composition), it is inevitable to require some changes on the
    model training code.

2. *Requires careful pipeline design*
    Until free-threaded (a.k.a no-GIL) Python becomes available, building efficient pipelines
    with :py:class:`spdl.dataloader.Pipeline` requires careful code review and expertise.
    It is easy to inadvertently mix functions that hold the GIL with those that release it,
    which can severely degrade performance.

    To achieve high throughput, one must use functions that are thread-safe and release GIL.
    Since SPDL comes with multimedia submodule which supports audio/video/image, and
    `OpenAI's tiktoken <https://github.com/openai/tiktoken>`_ and
    `HuggingFace's Tokenizers <https://github.com/huggingface/tokenizers>`_ † release GIL,
    we believe that the major modalities from popular ML models are covered.

    † The tokenizers is not thread-safe so it requires
    `a workaround <https://github.com/huggingface/tokenizers/issues/537#issuecomment-1372231603>`_.
    (note: You can use `thread local storage <https://docs.python.org/3/library/threading.html#thread-local-data>`_.)
