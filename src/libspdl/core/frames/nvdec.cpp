#include <libspdl/core/frames.h>

#include <libspdl/core/detail/ffmpeg/wrappers.h>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// NvDec - Video
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
NvDecFrames<media_type>::NvDecFrames(uint64_t id_, int format_)
    : id(id_), media_format(format_) {}

template <MediaType media_type>
const char* NvDecFrames<media_type>::get_media_format_name() const {
  return detail::get_media_format_name<media_type>(media_format);
}

template struct NvDecFrames<MediaType::Video>;
template struct NvDecFrames<MediaType::Image>;

template <MediaType media_type>
NvDecFramesPtr<media_type> clone(const NvDecFramesPtr<media_type>& src) {
  auto ret = std::make_unique<NvDecFrames<media_type>>(
      src->get_id(), src->media_format);
  ret->buffer = src->buffer;
  return ret;
}

template NvDecFramesPtr<MediaType::Video> clone(
    const NvDecFramesPtr<MediaType::Video>& src);
template NvDecFramesPtr<MediaType::Image> clone(
    const NvDecFramesPtr<MediaType::Image>& src);

} // namespace spdl::core
