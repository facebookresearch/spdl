message(STATUS "########################################")
message(STATUS "# Configuring SPDL")
message(STATUS "########################################")

# This adds the library directory to search path
if (SPDL_BUILD_PYTHON_BINDING)
  if (APPLE)
    set(CMAKE_INSTALL_RPATH "@loader_path")
  elseif (UNIX)
    set(CMAKE_INSTALL_RPATH $ORIGIN)
  endif ()
endif()

set(defs)
set(deps folly)
set(srcs
  libspdl/core/conversion/async.cpp
  libspdl/core/conversion/audio.cpp
  libspdl/core/conversion/nvdec.cpp
  libspdl/core/conversion/vision.cpp
  libspdl/core/detail/executor.cpp
  libspdl/core/detail/ffmpeg/conversion.cpp
  libspdl/core/detail/ffmpeg/ctx_utils.cpp
  libspdl/core/detail/ffmpeg/decoding.cpp
  libspdl/core/detail/ffmpeg/demuxing.cpp
  libspdl/core/detail/ffmpeg/filter_graph.cpp
  libspdl/core/detail/ffmpeg/wrappers.cpp
  libspdl/core/detail/logging.cpp
  libspdl/core/buffer/cpu.cpp
  libspdl/core/decoding/nvdec_async.cpp
  libspdl/core/decoding/ffmpeg_async.cpp
  libspdl/core/decoding/sleep.cpp
  libspdl/core/executor.cpp
  libspdl/core/demuxing.cpp
  libspdl/core/future.cpp
  libspdl/core/frames/ffmpeg.cpp
  libspdl/core/adaptor/adaptor.cpp
  libspdl/core/adaptor/bytes.cpp
  libspdl/core/adaptor/mmap.cpp
  libspdl/core/storage/cpu.cpp
  libspdl/core/packets.cpp
  libspdl/core/utils.cpp
  )

if (SPDL_USE_CUDA)
  list(APPEND defs SPDL_USE_CUDA)
  list(APPEND deps CUDA::cudart CUDA::cuda_driver)
  list(APPEND srcs
    libspdl/core/detail/cuda.cpp
    libspdl/core/buffer/cuda.cpp
    libspdl/core/storage/cpu.cpp
    libspdl/core/storage/cuda.cpp
    )
endif()

if (SPDL_USE_NVDEC)
  list(APPEND defs SPDL_USE_NVDEC)
  list(APPEND srcs
    libspdl/core/buffer/nvdec.cpp
    libspdl/core/frames/nvdec.cpp
    libspdl/core/detail/nvdec/converter.cpp
    libspdl/core/detail/nvdec/color_conversion.cu
    libspdl/core/detail/nvdec/decoder.cpp
    libspdl/core/detail/nvdec/decoding.cpp
    libspdl/core/detail/nvdec/utils.cpp
    libspdl/core/detail/nvdec/wrapper.cpp
    )
  list(APPEND deps nvcuvid-stub)
endif()

if(SPDL_ENABLE_TRACING)
  list(APPEND deps perfetto)
  list(APPEND defs SPDL_ENABLE_TRACING)
  list(APPEND srcs libspdl/core/detail/tracing.cpp)
endif()

if (SPDL_DEBUG_REFCOUNT)
  list(APPEND defs SPDL_DEBUG_REFCOUNT)
endif()

# install(
#   DIRECTORY libspdl
#   DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
#   FILES_MATCHING
#   PATTERN "*.h"
#   PATTERN "detail" EXCLUDE)

function(add_libspdl ffmpeg_version)
  set(name "spdl_ffmpeg${ffmpeg_version}")
  add_library("${name}" SHARED "${srcs}")
  target_compile_definitions("${name}" PRIVATE "${defs}")
  target_link_libraries("${name}" PUBLIC "${deps}" "ffmpeg${ffmpeg_version}")
  target_include_directories(
    "${name}"
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")

  install(
    TARGETS "${name}"
    EXPORT "${name}Targets"
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

endfunction()

add_libspdl(4)
add_libspdl(5)
add_libspdl(6)
add_libspdl(7)
