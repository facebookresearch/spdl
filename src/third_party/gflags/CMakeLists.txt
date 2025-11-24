message(STATUS "########################################")
message(STATUS "# Configuring gflags")
message(STATUS "########################################")

set(GFLAGS_NAMESPACE gflags)
set(GFLAGS_BUILD_STATIC_LIBS ON)
set(GFLAGS_BUILD_SHARED_LIBS OFF)
set(GFLAGS_BUILD_gflags_LIB ON)
set(GFLAGS_BUILD_gflags_nothreads_LIB OFF)

FetchContent_Declare(
  gflags
  URL https://github.com/gflags/gflags/archive/refs/tags/v2.2.0.tar.gz
  URL_HASH SHA256=466c36c6508a451734e4f4d76825cf9cd9b8716d2b70ef36479ae40f08271f88
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cache"
  OVERRIDE_FIND_PACKAGE
  )
FetchContent_MakeAvailable(gflags)
