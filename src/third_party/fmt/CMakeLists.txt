message(STATUS "########################################")
message(STATUS "# Configuring libfmt")
message(STATUS "########################################")

set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(
  fmt
  URL https://github.com/fmtlib/fmt/archive/refs/tags/11.0.2.tar.gz
  URL_HASH SHA256=6cb1e6d37bdcb756dbbe59be438790db409cdb4868c66e888d5df9f13f7c027f
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cache"
  )
FetchContent_MakeAvailable(fmt)
