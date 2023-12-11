message(STATUS "########################################")
message(STATUS "Configuring glog")
message(STATUS "########################################")

set(BUILD_SHARED_LIBS OFF)
set(WITH_GFLAGS OFF)

FetchContent_Declare(
  glog
  GIT_REPOSITORY https://github.com/mthrok/glog
  GIT_TAG v0.5.0-patch1
  )
FetchContent_MakeAvailable(glog)
