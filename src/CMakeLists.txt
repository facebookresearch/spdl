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
  libspdl/core/detail/executors.cpp
  libspdl/core/detail/ffmpeg/ctx_utils.cpp
  libspdl/core/detail/ffmpeg/decoding.cpp
  libspdl/core/detail/ffmpeg/filter_graph.cpp
  libspdl/core/detail/ffmpeg/wrappers.cpp
  libspdl/core/buffers.cpp
  libspdl/core/conversion.cpp
  libspdl/core/decoding.cpp
  libspdl/core/frames.cpp
  libspdl/core/interface/basic.cpp
  libspdl/core/interface/mmap.cpp
  libspdl/core/logging.cpp
  libspdl/core/storage.cpp
  libspdl/core/utils.cpp
  )

if (SPDL_USE_CUDA)
  list(APPEND defs SPDL_USE_CUDA)
  list(APPEND deps CUDA::cudart CUDA::cuda_driver)
  list(APPEND srcs libspdl/core/detail/cuda.cpp)
endif()

if (SPDL_DEBUG_REFCOUNT)
  list(APPEND defs SPDL_DEBUG_REFCOUNT)
endif()

install(
  DIRECTORY libspdl
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "detail" EXCLUDE)

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
