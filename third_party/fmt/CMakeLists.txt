message(STATUS "########################################")
message(STATUS "Configuring libfmt")
message(STATUS "########################################")

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 10.1.1
  )
FetchContent_MakeAvailable(fmt)
