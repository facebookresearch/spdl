message(STATUS "########################################")
message(STATUS "# Configuring SPDL")
message(STATUS "########################################")

set(CMAKE_CXX_VISIBILITY_PRESET default)

set(defs)
set(deps folly)
set(srcs
  detail/conversion.cpp
  detail/executors.cpp
  detail/ffmpeg/ctx_utils.cpp
  detail/ffmpeg/wrappers.cpp
  detail/ffmpeg/filter_graph.cpp
  detail/interface/mmap.cpp
  frames.cpp
  interface.cpp
  processors.cpp
  )

if (SPDL_USE_CUDA)
  list(APPEND defs SPDL_USE_CUDA)
  list(APPEND deps CUDA::cudart CUDA::cuda_driver)
endif()

if (SPDL_DEBUG_REFCOUNT)
  list(APPEND defs SPDL_DEBUG_REFCOUNT)
endif()

function(add_libspdl ffmpeg_version)
  set(name "spdl_ffmpeg${ffmpeg_version}")
  add_library("${name}_static" STATIC "${srcs}")
  target_compile_definitions("${name}_static" PRIVATE "${defs}")
  target_link_libraries("${name}_static" PUBLIC "${deps}" "ffmpeg${ffmpeg_version}")
  target_include_directories("${name}_static" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")

  install(
    TARGETS "${name}_static"
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    )

  add_library("${name}" SHARED "${srcs}")
  target_compile_definitions("${name}_static" PRIVATE "${defs}")
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
