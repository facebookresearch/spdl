message(STATUS "########################################")
message(STATUS "# Configuring SPDL")
message(STATUS "########################################")

set(CMAKE_CXX_VISIBILITY_PRESET default)

set(
  src
  detail/ffmpeg/cuda.cpp
  detail/ffmpeg/ctx_utils.cpp
  detail/ffmpeg/wrappers.cpp
  detail/ffmpeg/filter_graph.cpp
  detail/interface/mmap.cpp
  detail/executors.cpp
  buffer.cpp
  interface.cpp
  processors.cpp
  )

set(
  def)

if (SPDL_USE_CUDA)
  list(APPEND def SPDL_USE_CUDA)
endif()

function(add_libspdl ffmpeg_version)
  set(name "spdl_ffmpeg${ffmpeg_version}")
  add_library("${name}_static" STATIC "${src}")
  target_compile_definitions("${name}_static" PRIVATE "${def}")
  target_link_libraries("${name}_static" PUBLIC folly "ffmpeg${ffmpeg_version}")
  target_include_directories("${name}_static" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")

  install(
    TARGETS "${name}_static"
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    )

  add_library("${name}" SHARED "${src}")
  target_compile_definitions("${name}_static" PRIVATE "${def}")
  target_link_libraries("${name}" PUBLIC folly "ffmpeg${ffmpeg_version}")
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
