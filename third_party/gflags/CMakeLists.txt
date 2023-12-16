message(STATUS "########################################")
message(STATUS "Configuring gflags")
message(STATUS "########################################")

set(GFLAGS_BUILD_STATIC_LIBS ON)
set(GFLAGS_BUILD_SHARED_LIBS OFF)
set(GFLAGS_BUILD_gflags_LIB ON)
set(GFLAGS_BUILD_gflags_nothreads_LIB OFF)

FetchContent_Declare(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags
  GIT_TAG v2.2.0
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SUBBUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/subbuild"
  )
FetchContent_MakeAvailable(gflags)
