message(STATUS "########################################")
message(STATUS "# Configuring libdeflate")
message(STATUS "########################################")

set(LIBDEFLATE_BUILD_STATIC_LIB ON)
set(LIBDEFLATE_DECOMPRESSION_SUPPORT ON)
set(LIBDEFLATE_ZLIB_SUPPORT ON)

set(LIBDEFLATE_BUILD_SHARED_LIB OFF)
set(LIBDEFLATE_COMPRESSION_SUPPORT OFF)
set(LIBDEFLATE_GZIP_SUPPORT OFF)
set(LIBDEFLATE_BUILD_GZIP OFF)
set(LIBDEFLATE_BUILD_TESTS OFF)

FetchContent_Declare(
  libdeflate
  URL https://github.com/ebiggers/libdeflate/archive/refs/tags/v1.24.tar.gz
  URL_HASH SHA256=ad8d3723d0065c4723ab738be9723f2ff1cb0f1571e8bfcf0301ff9661f475e8
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cache"
  )
FetchContent_MakeAvailable(libdeflate)
