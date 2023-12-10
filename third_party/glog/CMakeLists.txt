message(STATUS "########################################")
message(STATUS "Configuring glog")
message(STATUS "########################################")

set(BUILD_SHARED_LIBS OFF)
set(WITH_GFLAGS OFF)

FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/google/glog
  GIT_TAG v0.5.0
  PATCH_COMMAND git am "${CMAKE_CURRENT_SOURCE_DIR}/glog_0.5.0.patch" || true
  )
FetchContent_MakeAvailable(glog)
