#include <libspdl/core/frames.h>

namespace spdl::core {
////////////////////////////////////////////////////////////////////////////////
// NvDec - Video
////////////////////////////////////////////////////////////////////////////////
NvDecVideoFrames::NvDecVideoFrames(uint64_t id_, MediaType type_, int format_)
    : id(id_), media_type(type_), media_format(format_) {}

bool NvDecVideoFrames::is_cuda() const {
  return true;
}

enum MediaType NvDecVideoFrames::get_media_type() const {
  return media_type;
}
} // namespace spdl::core
