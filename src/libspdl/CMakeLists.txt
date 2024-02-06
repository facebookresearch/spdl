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
  core/detail/executors.cpp
  core/detail/ffmpeg/ctx_utils.cpp
  core/detail/ffmpeg/decoding.cpp
  core/detail/ffmpeg/filter_graph.cpp
  core/detail/ffmpeg/wrappers.cpp
  core/buffers.cpp
  core/conversion.cpp
  core/decoding.cpp
  core/frames.cpp
  core/interface/basic.cpp
  core/interface/mmap.cpp
  core/logging.cpp
  core/storage.cpp
  core/utils.cpp
  )

if (SPDL_USE_CUDA)
  list(APPEND defs SPDL_USE_CUDA)
  list(APPEND deps CUDA::cudart CUDA::cuda_driver)
  list(APPEND srcs core/detail/cuda.cpp)
endif()

if (SPDL_DEBUG_REFCOUNT)
  list(APPEND defs SPDL_DEBUG_REFCOUNT)
endif()

function(add_libspdl ffmpeg_version)
  set(name "spdl_ffmpeg${ffmpeg_version}")
  add_library("${name}" SHARED "${srcs}")
  target_compile_definitions("${name}" PRIVATE "${defs}")
  target_link_libraries("${name}" PUBLIC "${deps}" "ffmpeg${ffmpeg_version}")
  target_include_directories("${name}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")

  install(
    TARGETS "${name}"
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    )

endfunction()

add_libspdl(4)
add_libspdl(5)
add_libspdl(6)
