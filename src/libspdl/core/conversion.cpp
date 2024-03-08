#include <libspdl/core/conversion.h>

#include <libspdl/core/detail/ffmpeg/conversion.h>
#include <libspdl/core/detail/logging.h>
#include <libspdl/core/types.h>

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
std::unique_ptr<Buffer> convert_audio_frames(
    const FFmpegAudioFrames* frames,
    const std::optional<int>& i) {
  return detail::convert_audio_frames(frames, i);
}

////////////////////////////////////////////////////////////////////////////////
// Video
////////////////////////////////////////////////////////////////////////////////
namespace {

template <bool single_image>
void check_consistency(const std::vector<AVFrame*>& frames) {
  auto numel = frames.size();
  if (numel == 0) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  if constexpr (single_image) {
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
          "Cannot convert the frames as the frames do not have the same size."));
    }
    if (static_cast<AVPixelFormat>(f->format) != pix_fmt) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames do not have the same pixel format."));
    }
  }
}

enum class TARGET { TO_CPU, TO_NATIVE };

using TARGET::TO_CPU;
using TARGET::TO_NATIVE;

template <TARGET target = TO_NATIVE, bool single_image = false>
std::unique_ptr<Buffer> convert_video(
    const std::vector<AVFrame*>& frames,
    const std::optional<int>& index) {
  check_consistency<single_image>(frames);
  bool is_cuda =
      static_cast<AVPixelFormat>(frames[0]->format) == AV_PIX_FMT_CUDA;
  if constexpr (target == TO_CPU) {
    if (is_cuda) {
      SPDL_FAIL("The input frames are not CPU frames.");
    }
    return detail::convert_video_frames_cpu(frames, index);
  }
  if constexpr (target == TO_NATIVE) {
    if (is_cuda) {
      return detail::convert_video_frames_cuda(frames, index);
    }
    return detail::convert_video_frames_cpu(frames, index);
  }
}
} // namespace

std::unique_ptr<Buffer> convert_video_frames(
    const FFmpegVideoFrames* frames,
    const std::optional<int>& index) {
  return convert_video<>(frames->frames, index);
}

std::unique_ptr<Buffer> convert_video_frames_to_cpu_buffer(
    const FFmpegVideoFrames* frames,
    const std::optional<int>& index) {
  return convert_video<TO_CPU>(frames->frames, index);
}

////////////////////////////////////////////////////////////////////////////////
// Image
////////////////////////////////////////////////////////////////////////////////
namespace {
template <TARGET t = TO_NATIVE>
std::unique_ptr<Buffer> convert_image(
    const std::vector<AVFrame*>& frames,
    const std::optional<int>& index) {
  auto buf = convert_video<t, /*single_image=*/true>(frames, index);
  buf->shape.erase(buf->shape.begin()); // Trim the first dim
  return buf;
}
} // namespace

std::unique_ptr<Buffer> convert_image_frames(
    const FFmpegImageFrames* frames,
    const std::optional<int>& index) {
  return convert_image<>(frames->frames, index);
}

std::unique_ptr<Buffer> convert_image_frames_to_cpu_buffer(
    const FFmpegImageFrames* frames,
    const std::optional<int>& index) {
  return convert_image<TO_CPU>(frames->frames, index);
}

////////////////////////////////////////////////////////////////////////////////
// Batch Image
////////////////////////////////////////////////////////////////////////////////
namespace {
std::vector<AVFrame*> merge_frames(
    const std::vector<FFmpegImageFrames*>& batch) {
  std::vector<AVFrame*> ret;
  ret.reserve(batch.size());
  for (auto& frame : batch) {
    if (frame->frames.size() != 1) {
      SPDL_FAIL_INTERNAL(
          "Unexpected number of frames are found in one of the image frames.");
    }
    ret.push_back(frame->frames[0]);
  }
  return ret;
}

} // namespace

std::unique_ptr<Buffer> convert_batch_image_frames(
    const std::vector<FFmpegImageFrames*>& batch,
    const std::optional<int>& index) {
  return convert_video<>(merge_frames(batch), index);
}

std::unique_ptr<Buffer> convert_batch_image_frames_to_cpu_buffer(
    const std::vector<FFmpegImageFrames*>& batch,
    const std::optional<int>& index) {
  return convert_video<TO_CPU>(merge_frames(batch), index);
}

std::shared_ptr<CUDABuffer2DPitch> convert_nvdec_video_frames(
    const NvDecVideoFrames* frames,
    const std::optional<int>& index) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  if (index.has_value()) {
    SPDL_FAIL_INTERNAL(
        "Fetching an index from NvDecVideoFrames is not supported.");
  }
  return frames->buffer;
#endif
}

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

void check_consistency(const std::vector<NvDecVideoFrames*>& frames) {
  auto numel = frames.size();
  if (numel == 0) {
    SPDL_FAIL("No frame to convert to buffer.");
  }
  auto pix_fmt = static_cast<AVPixelFormat>(frames[0]->media_format);
  auto shape = frames[0]->buffer->get_shape();
  for (auto* f : frames) {
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

std::shared_ptr<CUDABuffer2DPitch> convert_nvdec_batch_image_frames(
    const std::vector<NvDecVideoFrames*>& batch_frames,
    const std::optional<int>& index) {
#ifndef SPDL_USE_NVDEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  check_consistency(batch_frames);
  auto& buf0 = batch_frames[0]->buffer;

  detail::set_current_cuda_context(batch_frames[0]->buffer->p);
  auto ret = std::make_shared<CUDABuffer2DPitch>(batch_frames.size());
  ret->allocate(buf0->c, buf0->h, buf0->w, buf0->bpp, buf0->channel_last);

  cudaStream_t stream = 0;
  for (auto& frame : batch_frames) {
    CHECK_CUDA(
        cudaMemcpy2DAsync(
            ret->get_next_frame(),
            ret->pitch,
            (void*)frame->buffer->p,
            frame->buffer->pitch,
            frame->buffer->width_in_bytes,
            frame->buffer->h,
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
} // namespace spdl::core
