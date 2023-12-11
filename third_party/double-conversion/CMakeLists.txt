message(STATUS "########################################")
message(STATUS "Configuring double-conversion")
message(STATUS "########################################")

set(BUILD_SHARED_LIBS OFF)

FetchContent_Declare(
  double-conversion
  GIT_REPOSITORY https://github.com/google/double-conversion
  GIT_TAG v3.3.0
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SUBBUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/subbuild"
  )
FetchContent_MakeAvailable(double-conversion)
