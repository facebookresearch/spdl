#include <libspdl/core/conversion.h>

#include "libspdl/core/detail/ffmpeg/conversion.h"
#include "libspdl/core/detail/logging.h"
#include "libspdl/core/detail/tracing.h"

#ifdef SPDL_USE_NVCODEC
#include <libspdl/core/detail/cuda.h>
#endif

extern "C" {
#include <libavutil/frame.h>
}

namespace spdl::core {

template <MediaType media_type>
CUDABuffer2DPitchPtr convert_nvdec_frames(
    const NvDecFramesWrapperPtr<media_type> frames,
    const std::optional<int>& index) {
#ifndef SPDL_USE_NVCODEC
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

namespace {
#ifdef SPDL_USE_NVCODEC
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
  int device_index = f0->buffer->device_index;
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
    if (device_index != f->buffer->device_index) {
      SPDL_FAIL(fmt::format(
          "Cannot convert the frames as the frames are not on the same device index."));
    }
  }
}
#endif
} // namespace

CUDABuffer2DPitchPtr convert_nvdec_batch_image_frames(
    const std::vector<NvDecImageFramesWrapperPtr>& batch_frames,
    const std::optional<int>& index) {
#ifndef SPDL_USE_NVCODEC
  SPDL_FAIL("SPDL is not compiled with NVDEC support.");
#else
  TRACE_EVENT("decoding", "core::convert_nvdec_batch_image_frames");
  check_consistency(batch_frames);
  auto& buf0 = batch_frames[0]->get_frames_ref()->buffer;

  detail::set_cuda_primary_context(buf0->device_index);
  auto ret = std::make_shared<CUDABuffer2DPitch>(
      buf0->device_index, batch_frames.size());
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

} // namespace spdl::core
