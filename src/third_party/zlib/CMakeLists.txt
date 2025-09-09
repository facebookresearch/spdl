message(STATUS "########################################")
message(STATUS "# Configuring zlib")
message(STATUS "########################################")

set(BUILD_SHARED_LIBS OFF)
FetchContent_Declare(
  zlib
  URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
  URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
FetchContent_MakeAvailable(zlib)
