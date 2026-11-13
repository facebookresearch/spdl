message(STATUS "########################################")
message(STATUS "# Configuring libfmt")
message(STATUS "########################################")

set(BUILD_SHARED_LIBS OFF)
set(FMT_INSTALL ON)
# SPDL uses the header-based API, not fmt's optional C++ module target.
set(FMT_MODULE OFF)
FetchContent_Declare(
  fmt
  URL https://github.com/fmtlib/fmt/archive/refs/tags/12.2.0.tar.gz
  URL_HASH SHA256=8b852bb5aa6e7d8564f9e81394055395dd1d1936d38dfd3a17792a02bebd7af0
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cache"
  )
FetchContent_MakeAvailable(fmt)
