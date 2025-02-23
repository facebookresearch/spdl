message(STATUS "########################################")
message(STATUS "# Configuring glog")
message(STATUS "########################################")

# glog 0.5.0 requires 2.2.0, but won't fail in case it not found.
# but we do need, so just in case
find_package(gflags 2.2.0 REQUIRED)

set(BUILD_SHARED_LIBS OFF)
set(WITH_GFLAGS ON)
set(WITH_GTEST OFF)
set(gflags_NAMESPACE gflags)

FetchContent_Declare(
  glog
  URL https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz
  URL_HASH SHA256=eede71f28371bf39aa69b45de23b329d37214016e2055269b3b5e7cfd40b59f5
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
FetchContent_MakeAvailable(glog)
