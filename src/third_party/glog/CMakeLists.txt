message(STATUS "########################################")
message(STATUS "# Configuring glog")
message(STATUS "########################################")

set(BUILD_SHARED_LIBS OFF)
set(WITH_GFLAGS OFF)
set(WITH_GTEST OFF)

if (WIN32)
FetchContent_Declare(
  glog
  URL https://github.com/google/glog/archive/refs/tags/v0.5.0.tar.gz
  URL_HASH SHA256=eede71f28371bf39aa69b45de23b329d37214016e2055269b3b5e7cfd40b59f5
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  PATCH_COMMAND pwd && ls -lh && patch -p1 < "${CMAKE_CURRENT_SOURCE_DIR}/patch.diff"
  )
else()
FetchContent_Declare(
  glog
  URL https://github.com/google/glog/archive/refs/tags/v0.6.0.tar.gz
  URL_HASH SHA256=8a83bf982f37bb70825df71a9709fa90ea9f4447fb3c099e1d720a439d88bad6
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()
FetchContent_MakeAvailable(glog)
