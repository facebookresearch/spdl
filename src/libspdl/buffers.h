#pragma once

#include <libspdl/storage.h>

#include <memory>
#include <variant>
#include <vector>

namespace spdl {

// video buffer class to be exposed to python
struct VideoBuffer {
  using StorageVariants =
#ifdef SPDL_USE_CUDA
      std::variant<Storage, CUDAStorage>;
#else
      std::variant<Storage>;
#endif

  std::vector<size_t> shape;
  bool channel_last = false;

  std::shared_ptr<StorageVariants> storage;

  VideoBuffer(
      const std::vector<size_t> shape,
      bool channel_last,
      Storage&& storage);
#ifdef SPDL_USE_CUDA
  VideoBuffer(
      const std::vector<size_t> shape,
      bool channel_last,
      CUDAStorage&& storage);
#endif

  void* data();
  bool is_cuda() const;
};

VideoBuffer video_buffer(
    const std::vector<size_t> shape,
    bool channel_last = false);

#ifdef SPDL_USE_CUDA
VideoBuffer video_buffer_cuda(
    const std::vector<size_t> shape,
    CUstream stream,
    bool channel_last = false);
#endif
} // namespace spdl
