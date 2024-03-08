# Design Principles of LibSPDL

LibSPDL aims to be scalable and performant solution for ML dataloading.

- Scalable means that it supports large-scale training, and it aims to achieve this by being flexible enough to fit to various training system architectures.
- Performant means that the core functions are implemented in a way it minimizes the performance overhead and memory footprint.

This article looks at the overview of the dataloading system, and why such design was chosen. Please refer to [Understanding I/O Bottleneck](./bottleneck.md) for the basis of media processing. Also please note that the design of LibSPDL is highly inspired by [Accelerated Video Library from Tesla's AI Day 2022 presentation](https://www.youtube.com/live/ODSJsviD_SU?feature=shared&t=4933).

The core of LibSPDL is two sets of thread pools. The first one is for media decoding and the other is for other operations including demuxing and data acquisitions.

As we saw in [Understanding I/O Bottleneck](./bottleneck.md), media decoding involves tasks of different nature (and corresponding bottlenecks), that are I/O and compute. They have different constraints, and when attempting to achieving higher throughput, they respond differently to the increased load.

If the data are stored on local file system, data acquisition and demuxing is very fast. To match the speed of demuxing, one needs to spawn multiple of decoder threads and decode streams of packets concurrently.

On the other hand, if the data are transfered over the network, CPU cores are mostly on idle state. To increase the throughput of data acquisition, one can spawn multiple threads (within the network bandwidth) to fetch and demux multiple data concurrently.

Having dedicated thread pools for I/O and compute allows users to configure the system for better throughput that fits their compute environments.

The following figure illustrates this.

[The illustration of demuxing and decoding threads](../assets/thread_pools.png)

???+ note

    Internally, the data acquisition, demuxing and decoding are implemented using [C++ 20 coroutines](https://en.cppreference.com/w/cpp/language/coroutines). We use the excellent [`folly::coro`](https://github.com/facebook/folly/blob/main/folly/experimental/coro/README.md) library, which does all the heavy lifting, including but not limited to coroutine and thread pool management.
    
    The use of `folly::coro::Task`, `folly::coro::AsyncGenerator`, and `folly::CPUThreadPoolExecutor` makes it easy to express the decoding logic in asynchronous composable manner.
