#include <libspdl/core/frames.h>

#include <libspdl/core/detail/ffmpeg/logging.h>
#include <libspdl/core/detail/ffmpeg/wrappers.h>
#include <libspdl/core/detail/tracing.h>

extern "C" {
#include <libavutil/frame.h>
#include <libavutil/pixdesc.h>
}

namespace spdl::core {

////////////////////////////////////////////////////////////////////////////////
// FFmpeg Common
////////////////////////////////////////////////////////////////////////////////
FFmpegFrames::FFmpegFrames(uint64_t id_, MediaType type_)
    : id(id_), type(type_) {
  TRACE_EVENT(
      "decoding",
      "FFmpegFrames::FFmpegFrames",
      perfetto::Flow::ProcessScoped(id));
}

FFmpegFrames::FFmpegFrames(FFmpegFrames&& other) noexcept {
  *this = std::move(other);
}

FFmpegFrames& FFmpegFrames::operator=(FFmpegFrames&& other) noexcept {
  using std::swap;
  swap(id, other.id);
  swap(type, other.type);
  swap(frames, other.frames);
  return *this;
}

FFmpegFrames::~FFmpegFrames() {
  TRACE_EVENT(
      "decoding",
      "FFmpegFrames::~FFmpegFrames",
      perfetto::Flow::ProcessScoped(id));
  std::for_each(frames.begin(), frames.end(), [](AVFrame* p) {
    DEBUG_PRINT_AVFRAME_REFCOUNT(p);
    av_frame_unref(p);
    av_frame_free(&p);
  });
}

std::string FFmpegFrames::get_media_format() const {
  if (!frames.size()) {
    return "none";
  }
  switch (type) {
    case MediaType::Audio:
      return av_get_sample_fmt_name((AVSampleFormat)frames[0]->format);
    case MediaType::Video:
      return av_get_pix_fmt_name((AVPixelFormat)frames[0]->format);
  }
}

std::string FFmpegFrames::get_media_type() const {
  switch (type) {
    case MediaType::Audio:
      return "audio";
    case MediaType::Video:
      return "video";
  }
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Audio
////////////////////////////////////////////////////////////////////////////////
bool FFmpegAudioFrames::is_cuda() const {
  return false;
}

int FFmpegAudioFrames::get_sample_rate() const {
  return frames.size() ? frames[0]->sample_rate : -1;
}

int FFmpegAudioFrames::get_num_frames() const {
  int ret = 0;
  for (auto& f : frames) {
    ret += f->nb_samples;
  }
  return ret;
}

int FFmpegAudioFrames::get_num_channels() const {
  return frames.size() ? frames[0]->channels : -1;
}

////////////////////////////////////////////////////////////////////////////////
// FFmpeg - Video
////////////////////////////////////////////////////////////////////////////////
bool FFmpegVideoFrames::is_cuda() const {
  if (!frames.size()) {
    return false;
  }
  return static_cast<AVPixelFormat>(frames[0]->format) == AV_PIX_FMT_CUDA;
}

int FFmpegVideoFrames::get_num_frames() const {
  return frames.size();
}

int FFmpegVideoFrames::get_num_planes() const {
  return av_pix_fmt_count_planes((AVPixelFormat)frames[0]->format);
}

int FFmpegVideoFrames::get_width() const {
  return frames.size() ? frames[0]->width : -1;
}

int FFmpegVideoFrames::get_height() const {
  return frames.size() ? frames[0]->height : -1;
}

FFmpegVideoFrames FFmpegVideoFrames::slice(int start, int stop, int step)
    const {
  auto out = FFmpegVideoFrames{0, type};
  for (int i = start; i < stop; i += step) {
    AVFrame* dst = CHECK_AVALLOCATE(av_frame_alloc());
    CHECK_AVERROR(
        av_frame_ref(dst, frames[i]),
        "Failed to create a new reference to an AVFrame.");
    out.frames.push_back(dst);
  }
  return out;
}

} // namespace spdl::core
