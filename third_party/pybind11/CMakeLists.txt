message(STATUS "########################################")
message(STATUS "Configuring pybind11")
message(STATUS "########################################")

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11
  GIT_TAG  v2.11.1
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
  SUBBUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/subbuild"
  )
FetchContent_MakeAvailable(pybind11)
