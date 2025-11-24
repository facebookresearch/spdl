message(STATUS "########################################")
message(STATUS "# Configuring zlib")
message(STATUS "########################################")

set(ZLIB_BUILD_EXAMPLES OFF)
set(ZLIB_BUILD_TESTING OFF)
set(ZLIB_BUILD_SHARED OFF)
set(ZLIB_BUILD_STATIC ON)

# Using a development commit from 2025 Jan/Feb
# because CMake-related behavior was improved.
FetchContent_Declare(
  zlib
  URL https://github.com/madler/zlib/archive/5a82f71ed1dfc0bec044d9702463dbdf84ea3b71.tar.gz
  URL_HASH SHA256=54577b3e64db3d85268e909609faac1410d816376b037a1a415418f87f35814d
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cache"
  )
FetchContent_MakeAvailable(zlib)
