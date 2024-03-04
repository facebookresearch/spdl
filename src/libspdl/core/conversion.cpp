#include <libspdl/core/conversion.h>

#include <libspdl/core/detail/ffmpeg/conversion.h>
#include <libspdl/core/logging.h>
#include <libspdl/core/types.h>

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

#ifdef SPDL_USE_NVDEC
std::shared_ptr<CUDABuffer2DPitch> convert_nvdec_video_frames(
    const NvDecVideoFrames* frames,
    const std::optional<int>& index) {
  if (index.has_value()) {
    SPDL_FAIL_INTERNAL(
        "Fetching an index from NvDecVideoFrames is not supported.");
  }
  return frames->buffer;
}
#endif
} // namespace spdl::core
