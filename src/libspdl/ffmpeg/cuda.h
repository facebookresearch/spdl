struct AVBufferRef;

namespace spdl {
AVBufferRef* get_cuda_context(int index);
void clear_cuda_context_cache();
} // namespace spdl
