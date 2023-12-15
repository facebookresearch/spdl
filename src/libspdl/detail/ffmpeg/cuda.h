struct AVBufferRef;

namespace spdl::detail {
AVBufferRef* get_cuda_context(int index);
void clear_cuda_context_cache();
} // namespace spdl::detail
