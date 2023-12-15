extern "C" {
#include <libavutil/hwcontext.h>
}

#include <cassert>
#include <map>
#include <mutex>

#include <libspdl/detail/ffmpeg/logging.h>
#include <libspdl/detail/ffmpeg/wrappers.h>

namespace spdl::detail {

////////////////////////////////////////////////////////////////////////////////
// HardWare context
////////////////////////////////////////////////////////////////////////////////

namespace {

static std::mutex MUTEX;
static std::map<int, AVBufferRefPtr> CUDA_CONTEXT_CACHE;

} // namespace

AVBufferRef* get_cuda_context(int index) {
  std::lock_guard<std::mutex> lock(MUTEX);
  if (index == -1) {
    index = 0;
  }
  if (CUDA_CONTEXT_CACHE.count(index) == 0) {
    AVBufferRef* p = nullptr;
    CHECK_AVERROR(
        av_hwdevice_ctx_create(
            &p,
            AV_HWDEVICE_TYPE_CUDA,
            std::to_string(index).c_str(),
            nullptr,
            0),
        "Failed to create CUDA device context on device {}.",
        index);
    assert(p);
    CUDA_CONTEXT_CACHE.emplace(index, p);
    return p;
  }
  AVBufferRefPtr& buffer = CUDA_CONTEXT_CACHE.at(index);
  return buffer.get();
}

void clear_cuda_context_cache() {
  std::lock_guard<std::mutex> lock(MUTEX);
  CUDA_CONTEXT_CACHE.clear();
}

} // namespace spdl::detail
