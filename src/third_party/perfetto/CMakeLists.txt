message(STATUS "########################################")
message(STATUS "# Configuring perfetto")
message(STATUS "########################################")

FetchContent_Declare(
  perfetto_src
  URL https://github.com/google/perfetto/archive/refs/tags/v41.0.tar.gz
  URL_HASH SHA256=4c8fe8a609fcc77ca653ec85f387ab6c3a048fcd8df9275a1aa8087984b89db8
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
FetchContent_MakeAvailable(perfetto_src)

set(base_dir "${perfetto_src_SOURCE_DIR}")
add_library(perfetto STATIC "${base_dir}/sdk/perfetto.cc")
target_include_directories(perfetto SYSTEM PUBLIC "${base_dir}/sdk")
