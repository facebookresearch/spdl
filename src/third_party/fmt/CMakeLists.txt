message(STATUS "########################################")
message(STATUS "# Configuring libfmt")
message(STATUS "########################################")

set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(
  fmt
  URL https://github.com/fmtlib/fmt/archive/refs/tags/10.1.1.tar.gz
  URL_HASH SHA256=78b8c0a72b1c35e4443a7e308df52498252d1cefc2b08c9a97bc9ee6cfe61f8b
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cache"
  )
FetchContent_MakeAvailable(fmt)
