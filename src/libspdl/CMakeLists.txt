message(STATUS "########################################")
message(STATUS "Configuring SPDL")
message(STATUS "########################################")

set(
  src
  ffmpeg/cuda.cpp
  ffmpeg/utils.cpp
  ffmpeg/wrappers.cpp
  interface/interface.cpp
  interface/mmap.cpp
  processors.cpp
  )

add_library(spdl "${src}")
target_link_libraries(spdl PUBLIC folly ffmpeg6)
target_include_directories(spdl PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
