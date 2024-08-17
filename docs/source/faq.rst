Frequently Asked Questions
==========================

How to work around GIL?
-----------------------

In Python, GIL (Global Interpreter Lock) practically prevents the use of multi-threading, however extension modules that are written in low-level languages, such as C, C++ and Rust, can release GIL when executing operations that do not interact with Python interpreter.

Many libraries used for dataloading release the GIL. To name a few;

- Pillow
- OpenCV
- Decord
- tiktoken

Typically, the bottleneck of model training in loading and preprocessing the media data.
So even though there are still parts of pipelines that are constrained by GIL,
by taking advantage of preprocessing functions that release GIL,
we can achieve high throughput.

Why Async IO?
-------------

When training a model with large amount of data, the data are retrieved from remote locations. Network utilities often provide APIs based on Async I/O.

The Async I/O allows to easily build complex data preprocessing pipeline and execute them while automatically parallelizing parts of the pipline, achiving high throughput.

Synchronous operations that release GIL can be converted to async operations easily by running them in a thread pool. So by converting the synchronous preprocessing functions that release GIL into asynchronous operations, the entire data preprocessing pipeline can be executed in async event loop. The event loop handles the scheduling of data processing functions, and execute them concurrently.
