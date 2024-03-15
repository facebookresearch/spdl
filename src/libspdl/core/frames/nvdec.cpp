#include <libspdl/core/frames.h>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// NvDec - Video
////////////////////////////////////////////////////////////////////////////////
template <MediaType media_type>
NvDecFrames<media_type>::NvDecFrames(uint64_t id_, int format_)
    : id(id_), media_format(format_) {}

template struct NvDecFrames<MediaType::Video>;
template struct NvDecFrames<MediaType::Image>;
} // namespace spdl::core
