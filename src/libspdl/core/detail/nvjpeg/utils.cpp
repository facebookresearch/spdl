#include "libspdl/core/detail/nvjpeg/utils.h"
#include "libspdl/core/detail/tracing.h"

#include <folly/Singleton.h>
#include <folly/logging/xlog.h>

#include <set>

namespace spdl::core::detail {
////////////////////////////////////////////////////////////////////////////////
// NVJPEG library handle Singletons
////////////////////////////////////////////////////////////////////////////////
namespace {
struct nvjpegLibraryHandle {
  nvjpegHandle_t h;
  nvjpegLibraryHandle(nvjpegBackend_t be = NVJPEG_BACKEND_DEFAULT) {
    TRACE_EVENT("decoding", "nvjpegCreateEx");
    CHECK_NVJPEG(
        nvjpegCreateEx(be, nullptr, nullptr, NVJPEG_FLAGS_DEFAULT, &h),
        "Failed to create the NVJPEG library handle.");
  }
  ~nvjpegLibraryHandle() {
    TRACE_EVENT("decoding", "nvjpegDestroy");
    auto state = nvjpegDestroy(h);
    if (state != NVJPEG_STATUS_SUCCESS) {
      XLOG(CRITICAL) << "Failed to destroy the nvjpeg library handle: "
                     << detail::to_string(state);
    }
  }
  static std::shared_ptr<nvjpegLibraryHandle> get_nvjpeg(nvjpegBackend_t);
};

struct TagDefault {};
static folly::Singleton<nvjpegLibraryHandle, TagDefault> libnvjpeg_default;

struct TagHybrid {};
static folly::Singleton<nvjpegLibraryHandle, TagHybrid> libnvjpeg_hybrid([]() {
  return new nvjpegLibraryHandle(NVJPEG_BACKEND_HYBRID);
});

struct TagGpuHybrid {};
static folly::Singleton<nvjpegLibraryHandle, TagGpuHybrid> libnvjpeg_gpu_hybrid(
    []() { return new nvjpegLibraryHandle(NVJPEG_BACKEND_GPU_HYBRID); });

struct TagHardware {};
static folly::Singleton<nvjpegLibraryHandle, TagHardware> libnvjpeg_hardware(
    []() { return new nvjpegLibraryHandle(NVJPEG_BACKEND_HARDWARE); });

std::shared_ptr<nvjpegLibraryHandle> nvjpegLibraryHandle::get_nvjpeg(
    nvjpegBackend_t backend) {
  switch (backend) {
    case NVJPEG_BACKEND_DEFAULT:
      return libnvjpeg_default.try_get();
    case NVJPEG_BACKEND_HYBRID:
      return libnvjpeg_hybrid.try_get();
    case NVJPEG_BACKEND_GPU_HYBRID:
      return libnvjpeg_gpu_hybrid.try_get();
    case NVJPEG_BACKEND_HARDWARE:
      return libnvjpeg_hardware.try_get();
    default:
      SPDL_FAIL(fmt::format("Unexpected backend. {}", to_string(backend)));
  }
}
} // namespace

nvjpegHandle_t get_nvjpeg(nvjpegBackend_t backend) {
  auto lib = nvjpegLibraryHandle::get_nvjpeg(backend);
  if (!lib) {
    SPDL_FAIL("Failed to get the NVJPEG library handle.");
  }
  return lib->h;
}

////////////////////////////////////////////////////////////////////////////////
// nvjpegJpegState_t
////////////////////////////////////////////////////////////////////////////////
void nvjpeg_state_deleter::operator()(nvjpegJpegState* p) {
  TRACE_EVENT("decoding", "nvjpegJpegStateDestroy");
  if (!p) {
    auto status = nvjpegJpegStateDestroy(p);
    if (status != NVJPEG_STATUS_SUCCESS) {
      XLOG(WARN) << "Failed to destroy nvjpeg state: "
                 << detail::to_string(status);
    }
  }
};

nvjpegStatePtr get_nvjpeg_jpeg_state(nvjpegHandle_t nvjpeg_handle) {
  nvjpegJpegState_t jpeg_state = nullptr;
  TRACE_EVENT("decoding", "nvjpegJpegStateCreate");
  CHECK_NVJPEG(
      nvjpegJpegStateCreate(nvjpeg_handle, &jpeg_state),
      "Failed to create nvjpeg JPEG state.");
  return nvjpegStatePtr(jpeg_state);
}

////////////////////////////////////////////////////////////////////////////////
// Misc utils
////////////////////////////////////////////////////////////////////////////////
template <typename T>
std::set<std::string> get_keys(const std::map<std::string, T>& mapping) {
  std::set<std::string> keys;
  for (const auto& [k, _] : mapping) {
    keys.insert(k);
  }
  return keys;
}

nvjpegBackend_t get_nvjpeg_backend(const std::optional<std::string>& v) {
  if (!v) {
    return NVJPEG_BACKEND_DEFAULT;
  }
  const static std::map<std::string, nvjpegBackend_t> mapping = {
      {"default", NVJPEG_BACKEND_DEFAULT},
      {"hybrid", NVJPEG_BACKEND_HYBRID},
      {"gpu_hybrid", NVJPEG_BACKEND_GPU_HYBRID},
      {"hardware", NVJPEG_BACKEND_HARDWARE}};

  auto& be = *v;
  if (mapping.contains(be)) {
    return mapping.at(be);
  }
  SPDL_FAIL(fmt::format(
      "Unexpected backend: {}. Supported values are \"{}\"",
      be,
      fmt::join(get_keys(mapping), "\", \"")));
}

nvjpegOutputFormat_t get_nvjpeg_output_format(const std::string& f) {
  const static std::map<std::string, nvjpegOutputFormat_t> mapping = {
      {"rgb", NVJPEG_OUTPUT_RGB},
      {"bgr", NVJPEG_OUTPUT_BGR},
      {"rgb24", NVJPEG_OUTPUT_RGBI},
      {"bgr24", NVJPEG_OUTPUT_BGRI}};

  if (mapping.contains(f)) {
    return mapping.at(f);
  }
  SPDL_FAIL(fmt::format(
      "Unexpected pix_fmt: {}. Supported values are \"{}\"",
      f,
      fmt::join(get_keys(mapping), "\", \"")));
}

std::string to_string(nvjpegStatus_t s) {
  switch (s) {
    case NVJPEG_STATUS_SUCCESS:
      return "NVJPEG_STATUS_SUCCESS";
    case NVJPEG_STATUS_NOT_INITIALIZED:
      return "NVJPEG_STATUS_NOT_INITIALIZED";
    case NVJPEG_STATUS_INVALID_PARAMETER:
      return "NVJPEG_STATUS_INVALID_PARAMETER";
    case NVJPEG_STATUS_BAD_JPEG:
      return "NVJPEG_STATUS_BAD_JPEG";
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
      return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";
    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
      return "NVJPEG_STATUS_ALLOCATOR_FAILURE";
    case NVJPEG_STATUS_EXECUTION_FAILED:
      return "NVJPEG_STATUS_EXECUTION_FAILED";
    case NVJPEG_STATUS_ARCH_MISMATCH:
      return "NVJPEG_STATUS_ARCH_MISMATCH";
    case NVJPEG_STATUS_INTERNAL_ERROR:
      return "NVJPEG_STATUS_INTERNAL_ERROR";
    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
      return "NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED";
    case NVJPEG_STATUS_INCOMPLETE_BITSTREAM:
      return "NVJPEG_STATUS_INCOMPLETE_BITSTREAM";
    default:
      return fmt::format("Unknown nvjpegStatus_t code: {}", int(s));
  }
}

std::string to_string(nvjpegBackend_t b) {
  switch (b) {
    case NVJPEG_BACKEND_DEFAULT:
      return "NVJPEG_BACKEND_DEFAULT";
    case NVJPEG_BACKEND_HYBRID:
      return "NVJPEG_BACKEND_HYBRID";
    case NVJPEG_BACKEND_GPU_HYBRID:
      return "NVJPEG_BACKEND_GPU_HYBRID";
    case NVJPEG_BACKEND_HARDWARE:
      return "NVJPEG_BACKEND_HARDWARE";
    case NVJPEG_BACKEND_GPU_HYBRID_DEVICE:
      return "NVJPEG_BACKEND_GPU_HYBRID_DEVICE";
    case NVJPEG_BACKEND_HARDWARE_DEVICE:
      return "NVJPEG_BACKEND_HARDWARE_DEVICE";
#if NVJPEG_VER_MAJOR >= 13 || (NVJPEG_VER_MAJOR >= 12 && NVJPEG_VER_MINOR >= 2)
    case NVJPEG_BACKEND_LOSSLESS_JPEG:
      return "NVJPEG_BACKEND_LOSSLESS_JPEG";
#endif
    default:
      return fmt::format("Unknown nvjpegBackend_t value: {}", int(b));
  }
}

std::string to_string(nvjpegOutputFormat_t f) {
  switch (f) {
    case NVJPEG_OUTPUT_UNCHANGED:
      return "NVJPEG_OUTPUT_UNCHANGED";
    case NVJPEG_OUTPUT_YUV:
      return "NVJPEG_OUTPUT_YUV";
    case NVJPEG_OUTPUT_Y:
      return "NVJPEG_OUTPUT_Y";
    case NVJPEG_OUTPUT_RGB:
      return "NVJPEG_OUTPUT_RGB";
    case NVJPEG_OUTPUT_BGR:
      return "NVJPEG_OUTPUT_BGR";
    case NVJPEG_OUTPUT_RGBI:
      return "NVJPEG_OUTPUT_RGBI";
    case NVJPEG_OUTPUT_BGRI:
      return "NVJPEG_OUTPUT_BGRI";
#if NVJPEG_VER_MAJOR >= 13 || (NVJPEG_VER_MAJOR >= 12 && NVJPEG_VER_MINOR >= 2)
    case NVJPEG_OUTPUT_UNCHANGEDI_U16:
      return "NVJPEG_OUTPUT_UNCHANGEDI_U16";
#endif
    default:
      return fmt::format("Unknown nvjpegOutputFormat_t value: {}", int(f));
  }
}

} // namespace spdl::core::detail
