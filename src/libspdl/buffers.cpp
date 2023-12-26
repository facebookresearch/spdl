#include <libspdl/buffers.h>

namespace spdl {
namespace {

inline size_t prod(const std::vector<size_t>& shape) {
  size_t val = 1;
  for (auto& v : shape) {
    val *= v;
  }
  return val;
}

} // namespace

VideoBuffer::CPUStorage::CPUStorage(size_t size)
    : data(std::make_unique<uint8_t[]>(size)) {}

VideoBuffer::VideoBuffer(
    const std::vector<size_t> shape_,
    bool channel_last_,
    CPUStorage storage_)
    : shape(std::move(shape_)),
      channel_last(channel_last_),
      storage(std::move(storage_)) {}

template <class... Fs>
struct Overload : Fs... {
  using Fs::operator()...;
};
template <class... Fs>
Overload(Fs...) -> Overload<Fs...>;

void* VideoBuffer::data() {
  return std::visit(
      Overload{[](CPUStorage& s) { return static_cast<void*>(s.data.get()); }},
      storage);
}

VideoBuffer video_buffer(const std::vector<size_t> shape, bool channel_last) {
  return VideoBuffer{
      std::move(shape), channel_last, VideoBuffer::CPUStorage{prod(shape)}};
}

} // namespace spdl
