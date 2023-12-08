message(STATUS "########################################")
message(STATUS "Configuring SPDL")
message(STATUS "########################################")

set(
  src
  ffmpeg/cuda.cpp
  ffmpeg/ctx_utils.cpp
  ffmpeg/wrappers.cpp
  ffmpeg/filter_graph.cpp
  interface/interface.cpp
  interface/mmap.cpp
  processors.cpp
  )

function(add_libspdl ffmpeg_version)
  set(name "libspdl_ffmpeg${ffmpeg_version}")
  add_library("${name}" "${src}")
  target_link_libraries("${name}" PUBLIC folly "ffmpeg${ffmpeg_version}")
  target_include_directories("${name}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
endfunction()

add_libspdl(4)
add_libspdl(5)
add_libspdl(6)
