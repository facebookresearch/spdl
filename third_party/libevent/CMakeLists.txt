message(STATUS "########################################")
message(STATUS "Configuring libevent")
message(STATUS "########################################")

set(EVENT__LIBRARY_TYPE "STATIC")
set(EVENT__DISABLE_TESTS ON)
set(EVENT__DISABLE_REGRESS ON)
set(EVENT__DISABLE_SAMPLES ON)
set(EVENT__DISABLE_BENCHMARK ON)
set(EVENT__DISABLE_OPENSSL ON)
set(EVENT__DISABLE_THREAD_SUPPORT ON)

FetchContent_Declare(
  libevent
  GIT_REPOSITORY https://github.com/libevent/libevent
  GIT_TAG release-2.1.12-stable
  )
FetchContent_MakeAvailable(libevent)
