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
  detail/executors.cpp
  detail/ffmpeg/ctx_utils.cpp
  detail/ffmpeg/decoding.cpp
  detail/ffmpeg/filter_graph.cpp
  detail/ffmpeg/wrappers.cpp
  detail/interface/mmap.cpp
  buffers.cpp
  conversion.cpp
  decoding.cpp
  frames.cpp
  interface.cpp
  storage.cpp
  utils.cpp
  )

if (SPDL_USE_CUDA)
  list(APPEND defs SPDL_USE_CUDA)
  list(APPEND deps CUDA::cudart CUDA::cuda_driver)
  list(APPEND srcs detail/cuda.cpp)
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
