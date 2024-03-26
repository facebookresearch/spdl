# How C++ code is made async in Python

This note explains how libspdl make its primitive functions
implemented in C++ available to Python's async framework.

This note also serves as a guideline to make C++ code runs in
its own thread available to Python's async framework.

## Introduction

Python has powerful asynchronous frame, but due to the GIL,
it is not immediately applicable to CPU-heavy work loads.

[The official Python documentation](https://docs.python.org/3.12/library/asyncio-task.html#running-in-threads) has the following note:

> **Note:** Due to the [GIL](https://docs.python.org/3.12/glossary.html#term-GIL),
> `asyncio.to_thread()` can typically only be used to make IO-bound functions non-blocking.
> However, for extension modules that release the GIL or alternative Python
> implementations that don’t have one, `asyncio.to_thread()` can also be used
> for CPU-bound functions.

The libspdl is designed around the idea of having (thread pool) exeuctors dedicated
for CPU-bound operaitons (i.e. decoding) separate from I/O operations
(i.e. demuxing and data acquisition). So we have a way to offload the CPU-bound
operations to non-main threads, but how can we fit such functions to Python's
asynchronous paradigm?

Although I thought this would be a popular implementation pattern, when I searched
this topic on the web, there was no apparent tutorial or question/answer to do this.

## Converting synchronous function to asynchronous function

Converting a blocking function to asynchronous function is a popular approach and
there are many questions on how to do and answers found on the web.

 - https://stackoverflow.com/questions/43241221/how-can-i-wrap-a-synchronous-function-in-an-async-coroutine

The popular answer to these question is to use
[`loop.run_in_executor`](https://docs.python.org/3.12/library/asyncio-eventloop.html#asyncio.loop.run_in_executor).

However, this spawns thread pool executor in Python side, on top of the thread pools
we have in C++ code, which feels like an additional overhead.

How can we just start our own async code in the main loop and wait for the result
without blocking the loop?

## Two kinds of Futures

The implementation of [`loop.run_in_executor`](https://github.com/python/cpython/blob/eebea7e515462b503632ada74923ec3246599c9c/Lib/asyncio/base_events.py#L883-L897) consists of three steps;

1. Create a `ThreadPoolExecutor`.
2. Submit the function to the executor and obtain `concurrent.futures.Future` object.
3. Wrap the `concurrent.futures.Future` object with `asyncio.Future`, using `asyncio.wrap_future` fubction.

If we can represent the state of our code running in C++ thread with `concurrent.futures.Future`, then we can wrap it with [`asyncio.wrap_future`](https://docs.python.org/3/library/asyncio-future.html#asyncio.wrap_future), which is a documented API.

If we further look into the implementation of [`concurrent.futures.ThreadPoolExecutor.submit`](https://github.com/python/cpython/blob/eebea7e515462b503632ada74923ec3246599c9c/Lib/concurrent/futures/thread.py#L164-L181), the essence of the manipulation of `concurrent.futures.Future` is found in [`_WorkItem.fun`](https://github.com/python/cpython/blob/eebea7e515462b503632ada74923ec3246599c9c/Lib/concurrent/futures/thread.py#L53-L64) method.

It is as simple as calling [`Future.set_result`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.set_result) if the operation succeeds and [`Future.set_exception`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future.set_exception) otherwise.

## Adding callbacks to C++ code

It seems that if we can call `Future.set_result` and `Future.set_exception` from our C++ code,
that will make the code runnable in Python asynchronous loop.

How can we call Python function/method from C++ code?

It turned out that PyBind11 can handle this without introducing Python dependency in C++ codebase.

First, we wrap the original C++ code so that it accepts two callbacks that corresponds to `Future.set_result` and `Future.set_exception`.

Here, let's say we have a function that does some heavy calculation and its result is `int` type.

```C++
int calc();
```

and this function is executed in a background thread.
(In libspdl, this is implemented with `folly`'s coroutines and executors)

```C++
std::thread run_in_thread() {
    std::thread t{[](){
        // Let's pretend that the result is somehow magically retrieved
        calc();
    }};
    return t;
}
```

We adds two call back functions to this.

`set_result()` callback receives the result of the calculation, and
`notify_exception()` is just a notification of exception.

```C++
std::thread run_in_thread(
  std::function<void(int)> set_result,
  std::function<void()> notify_exception
) {
    std::thread t{[](){
        try {
            int result = calc();
            set_result(result);
        catch (...) {
            notify_exception();

            // Some code to store the exception somewhere
            // until it's retrieved.
            store_exception();
        }
    }};
    return t;
}
```

```C++
// Retrieve the exception stored by `run_in_thread`
void retrieve_exception();
```

PyBind11 can bind such function without a hussle, but the real magic happens
at runtime. We will call the above function like the following

```python

class _CustomError(Exception):
    pass


async def calc():
    future = concurrent.futures.Future()

    def notify_exception():
        future.set_exception(_CustomError(""))

    t = run_in_thread(future.set_result, notify_exception)

    try:
        return await asyncio.futures.wrap_future(future)
    except _CustomError:
        pass

    raise retrieve_exception();
```

The instance method `future.set_result` is dynamically converted to
`std::function<void(int)>` type and the C++ code can call it without problem.

Propagating the exception is a bit tricky as IIUC, Python (or PyBind11)
exception is not a class or instance at its C implementation level.

Instead we just introduce a helper function of signature `() -> None` and
invoke it from C++. This will make the awaiting on `asyncio.Future` fail and
move the execution to the next block, where we can retrieve and re-throw the
exception from C++ (which PyBind11 will translate to some `RuntimeError`.)

## LibSPDL implementation

In the above example, we pretend that there is a some nice feature to store
and retrieve the exception happened during the execution of the job.

In libspdl, the jobs are implemented with `folly`'s coroutine (such as
`folly::coro::Task<T>` and `folly::coro::AsyncGenerator<T>`), and are executed
in `folly::CPUThreadPoolExecutor`.

The bound code returns `folly::SemiFuture<T>`, which also serves as a handler
to the execution results and exceptions.

```C++
folly::SemiFuture<int> run_in_bg(
  std::function<void(int)> set_result,
  std::function<void()> notify_exception
) {
    auto task = folly::coro::co_invoke([](){
        try {
            set_result(calc());
        } except (...) {
            notify_exception();
            throw; // rethrow, this will be recorded in SemiFuture
        }
    });
    return std::move(task).scheduleOn(executor).start();
}
```

So that the part about the exception retrieval looks like the following.

```python
async def calc():
    future = concurrent.futures.Future()

    def notify_exception():
        future.set_exception(_CustomError(""))

    semi_future = run_in_thread(future.set_result, notify_exception)

    try:
        return await asyncio.futures.wrap_future(future)
    except _CustomError:
        pass

    // This part of the code is only reached when background job failed

    semi_future.get() // will throw
```

## Adding cancellation

`async def` syntax defines a coroutine. Wrapping it with `asyncio.Task`
give an opportunity to insert cancellation logic to it.

```python
loop = asyncio.get_running_loop()
task = loop.create_task(calc())
task.cancel()
```

If an instance of `asyncio.Future` object is awaited when a task wrapping
the future is cancelle, it raises `asyncio.CancelledError`.
`asyncio.CancelledError` does not inherit `Exception` but `BaseException`
(Python 3.8+). Therefore, so as to handle the cancellation, we need to
catch `asyncio.CancelledError` when awaiting the `asyncio.Future` object.

```python
    semi_future = run_in_thread(future.set_result, notify_exception)

    try:
        return await asyncio.futures.wrap_future(future)
    except _CustomError:
        pass
    except asyncio.CancelledError as e:
        // call cancel on C++ side
        semi_future.cancel()
        try:
            // Wait again until the cancellation is propagated
            await asyncio.futures.wrap_future(future)
        except _CustomError:
            pass
        // Raise the original exception.
        // It is not recommended to swallow the cancellation.
        // https://docs.python.org/3.12/library/asyncio-task.html#task-cancellation
        raise e

    semi_future.get()
```

!!! note

    The object returned from `run_in_thread` has been changed to a structure
    that holds `folly::SemiFuture<int>` and `folly::CancellationSource`, but
    such detail is omitted here.
