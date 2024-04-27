message(STATUS "########################################")
message(STATUS "# Configuring double-conversion")
message(STATUS "########################################")

set(BUILD_SHARED_LIBS OFF)

FetchContent_Declare(
  double-conversion
  URL https://github.com/google/double-conversion/archive/refs/tags/v3.3.0.tar.gz
  URL_HASH SHA256=04ec44461850abbf33824da84978043b22554896b552c5fd11a9c5ae4b4d296e
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
FetchContent_MakeAvailable(double-conversion)
