#include <libspdl/core/conversion.h>

#include <libspdl/core/types.h>

#include "libspdl/core/detail/executor.h"
#include "libspdl/core/detail/ffmpeg/conversion.h"
#include "libspdl/core/detail/future.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#ifdef SPDL_USE_NVDEC
#include <libspdl/core/detail/cuda.h>
#endif

#include <fmt/core.h>
#include <folly/logging/xlog.h>

#include <cassert>

extern "C" {
#include <libavutil/frame.h>
}

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// Audio
////////////////////////////////////////////////////////////////////////////////
BufferPtr convert_audio_frames(
    const FFmpegAudioFramesWrapperPtr frames,
    const std::optional<int>& i) {
  TRACE_EVENT(
      "decoding",
      "core::convert_audio_frames",
      perfetto::Flow::ProcessScoped(frames->get_id()));
  return detail::convert_audio_frames(frames->get_frames_ref().get(), i);
}

////////////////////////////////////////////////////////////////////////////////
// Video
////////////////////////////////////////////////////////////////////////////////
namespace {

template <MediaType media_type>
void check_consistency(const std::vector<AVFrame*>& frames) requires(
    media_type != MediaType::Audio) {
  auto numel = frames.size();
  if (numel == 0) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  if constexpr (media_type == MediaType::Image) {
    if (numel != 1) {
      SPDL_FAIL_INTERNAL(fmt::format(
          "There must be exactly one frame to convert to buffer. Found: {}",
          numel));
    }
  }
  auto pix_fmt = static_cast<AVPixelFormat>(frames[0]->format);
  int height = frames[0]->height, width = frames[0]->width;
  for (auto* f : frames) {
    if (f->height != height || f->width != width) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames do not have the same size. "
          "Reference WxH = {}x{}, found {}x{}.",
          height,
          width,
          f->height,
          f->width));
    }
    if (static_cast<AVPixelFormat>(f->format) != pix_fmt) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames do not have the same pixel format."));
    }
  }
}

template <MediaType media_type, bool cpu_only>
BufferPtr convert_video(
    const std::vector<AVFrame*>& frames,
    const std::optional<int>& index) requires(media_type != MediaType::Audio) {
  check_consistency<media_type>(frames);
  bool is_cuda =
      static_cast<AVPixelFormat>(frames[0]->format) == AV_PIX_FMT_CUDA;
  if (is_cuda) {
    if constexpr (cpu_only) {
      SPDL_FAIL("The input frames are not CPU frames.");
    }
    return detail::convert_video_frames_cuda(frames, index);
  }
  auto buf = detail::convert_video_frames_cpu(frames, index);
  if constexpr (media_type == MediaType::Image) {
    buf->shape.erase(buf->shape.begin()); // Trim the first dim
  }
  return buf;
}
} // namespace

template <MediaType media_type, bool cpu_only>
BufferPtr convert_visual_frames(
    const FFmpegFramesWrapperPtr<media_type> frames,
    const std::optional<int>& index) requires(media_type != MediaType::Audio) {
  TRACE_EVENT(
      "decoding",
      "core::convert_visual_frames",
      perfetto::Flow::ProcessScoped(frames->get_id()));
  return convert_video<media_type, cpu_only>(
      frames->get_frames_ref()->get_frames(), index);
}

template BufferPtr convert_visual_frames<MediaType::Video, true>(
    const FFmpegFramesWrapperPtr<MediaType::Video> frames,
    const std::optional<int>& index);

template BufferPtr convert_visual_frames<MediaType::Video, false>(
    const FFmpegFramesWrapperPtr<MediaType::Video> frames,
    const std::optional<int>& index);

template BufferPtr convert_visual_frames<MediaType::Image, true>(
    const FFmpegFramesWrapperPtr<MediaType::Image> frames,
    const std::optional<int>& index);

template BufferPtr convert_visual_frames<MediaType::Image, false>(
    const FFmpegFramesWrapperPtr<MediaType::Image> frames,
    const std::optional<int>& index);

////////////////////////////////////////////////////////////////////////////////
// Batch Image
////////////////////////////////////////////////////////////////////////////////
namespace {
std::vector<AVFrame*> merge_frames(
    const std::vector<FFmpegImageFramesWrapperPtr>& batch) {
  std::vector<AVFrame*> ret;
  ret.reserve(batch.size());
  for (auto& frame : batch) {
    auto& ref = frame->get_frames_ref();
    if (ref->get_num_frames() != 1) {
      SPDL_FAIL_INTERNAL(
          "Unexpected number of frames are found in one of the image frames.");
    }
    ret.push_back(ref->get_frames()[0]);
  }
  return ret;
}

} // namespace

template <bool cpu_only>
BufferPtr convert_batch_image_frames(
    const std::vector<FFmpegImageFramesWrapperPtr>& batch,
    const std::optional<int>& index) {
  TRACE_EVENT("decoding", "core::convert_batch_image_frames");
  return convert_video<MediaType::Video, false>(merge_frames(batch), index);
}

template BufferPtr convert_batch_image_frames<true>(
    const std::vector<FFmpegImageFramesWrapperPtr>& batch,
    const std::optional<int>& index);

template BufferPtr convert_batch_image_frames<false>(
    const std::vector<FFmpegImageFramesWrapperPtr>& batch,
    const std::optional<int>& index);

template <MediaType media_type>
CUDABuffer2DPitchPtr convert_nvdec_frames(
    const NvDecFramesWrapperPtr<media_type> frames,
    const std::optional<int>& index) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  TRACE_EVENT(
      "decoding",
      "core::convert_nvdec_frames",
      perfetto::Flow::ProcessScoped(frames->get_id()));
  if (index.has_value()) {
    SPDL_FAIL_INTERNAL(
        "Fetching an index from NvDecVideoFrames is not supported.");
  }
  auto ret = frames->get_frames_ref()->buffer;
  if (!ret) {
    SPDL_FAIL("Attempted to convert an empty NvDecVideoFrames.");
  }
  return ret;
#endif
}

template CUDABuffer2DPitchPtr convert_nvdec_frames<MediaType::Image>(
    const NvDecFramesWrapperPtr<MediaType::Image> frames,
    const std::optional<int>& index);

template CUDABuffer2DPitchPtr convert_nvdec_frames<MediaType::Video>(
    const NvDecFramesWrapperPtr<MediaType::Video> frames,
    const std::optional<int>& index);

#ifdef SPDL_USE_NVDEC
namespace {
bool same_shape(const std::vector<size_t>& a, const std::vector<size_t>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

template <MediaType media_type>
void check_consistency(
    const std::vector<NvDecFramesWrapperPtr<media_type>>& frames) {
  auto numel = frames.size();
  if (numel == 0) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  auto& f0 = frames[0]->get_frames_ref();
  auto pix_fmt = static_cast<AVPixelFormat>(f0->media_format);
  auto shape = f0->buffer->get_shape();
  for (auto& frm : frames) {
    auto& f = frm->get_frames_ref();
    if (auto shape_ = f->buffer->get_shape(); !same_shape(shape, shape_)) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames do not have the same size."));
    }
    if (static_cast<AVPixelFormat>(f->media_format) != pix_fmt) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames do not have the same pixel format."));
    }
  }
}
} // namespace
#endif

CUDABuffer2DPitchPtr convert_nvdec_batch_image_frames(
    const std::vector<NvDecImageFramesWrapperPtr>& batch_frames,
    const std::optional<int>& index) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  TRACE_EVENT("decoding", "core::convert_nvdec_batch_image_frames");
  check_consistency(batch_frames);
  auto& buf0 = batch_frames[0]->get_frames_ref()->buffer;

  detail::set_current_cuda_context(buf0->p);
  auto ret = std::make_shared<CUDABuffer2DPitch>(batch_frames.size());
  ret->allocate(buf0->c, buf0->h, buf0->w, buf0->bpp, buf0->channel_last);

  cudaStream_t stream = 0;
  for (auto& frame : batch_frames) {
    auto& buf = frame->get_frames_ref()->buffer;
    CHECK_CUDA(
        cudaMemcpy2DAsync(
            ret->get_next_frame(),
            ret->pitch,
            (void*)buf->p,
            buf->pitch,
            buf->width_in_bytes,
            buf->h,
            cudaMemcpyDefault,
            stream),
        "Failed to launch cudaMemcpy2DAsync.");
    ret->n += 1;
  }
  CHECK_CUDA(
      cudaStreamSynchronize(stream),
      "Failed to synchronize the stream after copying the data.");
  return ret;
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Async - FFmpeg
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type, bool cpu_only>
FuturePtr async_convert_frames(
    std::function<void(BufferPtr)> set_result,
    std::function<void()> notify_exception,
    FFmpegFramesWrapperPtr<media_type> frames,
    const std::optional<int>& index,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](FFmpegFramesWrapperPtr<media_type>&& frm)
          -> folly::coro::Task<BufferPtr> {
        if constexpr (media_type == MediaType::Audio) {
          co_return convert_audio_frames(std::move(frm), index);
        } else {
          co_return convert_visual_frames<media_type>(std::move(frm), index);
        }
      },
      // Pass the ownership of FramePtr to executor thread, so that it is
      // deallocated there, instead of the main thread.
      wrap<media_type, FFmpegFramesPtr>(frames->unwrap()));
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

template FuturePtr async_convert_frames<MediaType::Audio, true>(
    std::function<void(BufferPtr)> set_result,
    std::function<void()> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Audio> frames,
    const std::optional<int>& index,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_frames<MediaType::Video, true>(
    std::function<void(BufferPtr)> set_result,
    std::function<void()> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Video> frames,
    const std::optional<int>& index,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_frames<MediaType::Video, false>(
    std::function<void(BufferPtr)> set_result,
    std::function<void()> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Video> frames,
    const std::optional<int>& index,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_frames<MediaType::Image, true>(
    std::function<void(BufferPtr)> set_result,
    std::function<void()> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Image> frames,
    const std::optional<int>& index,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_frames<MediaType::Image, false>(
    std::function<void(BufferPtr)> set_result,
    std::function<void()> notify_exception,
    FFmpegFramesWrapperPtr<MediaType::Image> frames,
    const std::optional<int>& index,
    ThreadPoolExecutorPtr executor);

////////////////////////////////////////////////////////////////////////////////
// Async - NVDEC
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
FuturePtr async_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void()> notify_exception,
    NvDecFramesWrapperPtr<media_type> frames,
    const std::optional<int>& index,
    ThreadPoolExecutorPtr executor) {
  auto task =
      folly::coro::co_invoke([=]() -> folly::coro::Task<CUDABuffer2DPitchPtr> {
        // CUDABuffer2DPitchPtr uses shared_ptr, and the CUDA buffer is shared
        // among Frames class and Buffer class, so unlike CPU case, we do not
        // need to deallocate here.
        co_return convert_nvdec_frames<media_type>(frames, index);
      });
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

template FuturePtr async_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void()> notify_exception,
    NvDecFramesWrapperPtr<MediaType::Video> frames,
    const std::optional<int>& index,
    ThreadPoolExecutorPtr executor);

template FuturePtr async_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void()> notify_exception,
    NvDecFramesWrapperPtr<MediaType::Image> frames,
    const std::optional<int>& index,
    ThreadPoolExecutorPtr executor);

////////////////////////////////////////////////////////////////////////////////
// Async batch conversion
////////////////////////////////////////////////////////////////////////////////

namespace {
std::vector<FFmpegImageFramesWrapperPtr> rewrap(
    std::vector<FFmpegImageFramesWrapperPtr>&& frames) {
  std::vector<FFmpegImageFramesWrapperPtr> ret;
  ret.reserve(frames.size());
  for (auto& frame : frames) {
    ret.emplace_back(wrap<MediaType::Image, FFmpegFramesPtr>(frame->unwrap()));
  }
  return ret;
}
} // namespace

FuturePtr async_batch_convert_frames(
    std::function<void(BufferPtr)> set_result,
    std::function<void()> notify_exception,
    std::vector<FFmpegImageFramesWrapperPtr> frames,
    const std::optional<int>& index,
    ThreadPoolExecutorPtr executor) {
  auto task = folly::coro::co_invoke(
      [=](std::vector<FFmpegImageFramesWrapperPtr>&& frms)
          -> folly::coro::Task<BufferPtr> {
        co_return convert_batch_image_frames(frms, index);
      },
      // Pass the ownership of FramePtrs to executor thread, so that they are
      // deallocated there, instead of the main thread.
      rewrap(std::move(frames)));
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}

FuturePtr async_batch_convert_nvdec_frames(
    std::function<void(CUDABuffer2DPitchPtr)> set_result,
    std::function<void()> notify_exception,
    std::vector<NvDecImageFramesWrapperPtr> frames,
    const std::optional<int>& index,
    ThreadPoolExecutorPtr executor) {
  auto task =
      folly::coro::co_invoke([=]() -> folly::coro::Task<CUDABuffer2DPitchPtr> {
        co_return convert_nvdec_batch_image_frames(frames, index);
      });
  return detail::execute_task_with_callback(
      std::move(task),
      set_result,
      notify_exception,
      detail::get_demux_executor_high_prio(executor));
}
} // namespace spdl::core
