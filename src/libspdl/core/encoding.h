#include <libspdl/core/types.h>

#include <string>
#include <vector>

namespace spdl::core {

void encode_image(
    std::string uri,
    void* data,
    std::vector<size_t> shape,
    int bit_depth,
    const std::string& src_pix_fmt,
    const std::optional<EncodeConfig>& enc_cfg);

} // namespace spdl::core
