#include <libspdl/core/conversion.h>

#include "libspdl/core/detail/ffmpeg/conversion.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#include <fmt/core.h>

extern "C" {
#include <libavutil/frame.h>
}

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// Video/Image
////////////////////////////////////////////////////////////////////////////////
namespace {

template <MediaType media_type>
void check_consistency(const std::vector<AVFrame*>& frames)
  requires(media_type != MediaType::Audio)
{
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
    const std::optional<int>& cuda_device_index)
  requires(media_type != MediaType::Audio)
{
  check_consistency<media_type>(frames);
  bool is_cuda =
      static_cast<AVPixelFormat>(frames[0]->format) == AV_PIX_FMT_CUDA;
  if (is_cuda) {
    if constexpr (cpu_only) {
      SPDL_FAIL("The input frames are not CPU frames.");
    }
    if (!cuda_device_index) {
      SPDL_FAIL_INTERNAL(
          "CUDA frames are found, but cuda device index is not set.");
    }
    return detail::convert_video_frames_cuda(frames, cuda_device_index.value());
  }
  auto buf = detail::convert_video_frames_cpu(frames);
  if constexpr (media_type == MediaType::Image) {
    buf->shape.erase(buf->shape.begin()); // Trim the first dim
  }
  return buf;
}
} // namespace

template <MediaType media_type, bool cpu_only>
BufferPtr convert_vision_frames(const FFmpegFramesWrapperPtr<media_type> frames)
  requires(media_type != MediaType::Audio)
{
  TRACE_EVENT(
      "decoding",
      "core::convert_vision_frames",
      perfetto::Flow::ProcessScoped(frames->get_id()));
  auto& frames_ref = frames->get_frames_ref();
  return convert_video<media_type, cpu_only>(
      frames_ref->get_frames(), frames_ref->cuda_device_index);
}

template BufferPtr convert_vision_frames<MediaType::Video, true>(
    const FFmpegFramesWrapperPtr<MediaType::Video> frames);

template BufferPtr convert_vision_frames<MediaType::Video, false>(
    const FFmpegFramesWrapperPtr<MediaType::Video> frames);

template BufferPtr convert_vision_frames<MediaType::Image, true>(
    const FFmpegFramesWrapperPtr<MediaType::Image> frames);

template BufferPtr convert_vision_frames<MediaType::Image, false>(
    const FFmpegFramesWrapperPtr<MediaType::Image> frames);

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
    const std::vector<FFmpegImageFramesWrapperPtr>& batch) {
  TRACE_EVENT("decoding", "core::convert_batch_image_frames");
  return convert_video<MediaType::Video, false>(
      merge_frames(batch), batch[0]->get_frames_ref()->cuda_device_index);
}

template BufferPtr convert_batch_image_frames<true>(
    const std::vector<FFmpegImageFramesWrapperPtr>& batch);

template BufferPtr convert_batch_image_frames<false>(
    const std::vector<FFmpegImageFramesWrapperPtr>& batch);

} // namespace spdl::core
