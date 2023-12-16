message(STATUS "########################################")
message(STATUS "Configuring SPDL")
message(STATUS "########################################")

set(CMAKE_CXX_VISIBILITY_PRESET default)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-nullability-completeness")

set(
  src
  detail/ffmpeg/cuda.cpp
  detail/ffmpeg/ctx_utils.cpp
  detail/ffmpeg/wrappers.cpp
  detail/ffmpeg/filter_graph.cpp
  detail/interface/mmap.cpp
  interface.cpp
  processors.cpp
  )

function(add_libspdl ffmpeg_version)
  set(name "spdl_ffmpeg${ffmpeg_version}")
  add_library("${name}" SHARED "${src}")
  target_link_libraries("${name}" PUBLIC folly "ffmpeg${ffmpeg_version}")
  target_include_directories("${name}" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")

  add_library("${name}_static" STATIC "${src}")
  target_link_libraries("${name}_static" PUBLIC folly "ffmpeg${ffmpeg_version}")
  target_include_directories("${name}_static" PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")

  install(
    TARGETS "${name}" "${name}_static"
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
    )
endfunction()

add_libspdl(4)
add_libspdl(5)
add_libspdl(6)
