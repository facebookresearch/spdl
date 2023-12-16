message(STATUS "########################################")
message(STATUS "# Configuring Boost")
message(STATUS "########################################")

set(BOOST_INCLUDE_LIBRARIES
  algorithm
  array
  assert
  bind
  exception
  config
  concept_check
  conversion
  container_hash
  context
  core
  describe
  detail
  function
  function_types
  functional
  fusion
  integer
  intrusive
  io
  iterator
  move
  mp11
  mpl
  optional
  pool
  predef
  preprocessor
  range
  regex
  unordered
  static_assert
  smart_ptr
  tuple
  type_index
  typeof
  utility
  variant
  winapi
  throw_exception
  type_traits
  )

set(BOOST_SKIP_INSTALL_RULES OFF)

message(STATUS "Downloading & extracting boost source... (This may take a while.)" )

FetchContent_Declare(
  boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
  URL_HASH SHA256=2e64e5d79a738d0fa6fb546c6e5c2bd28f88d268a2a080546f74e5ff98f29d0e
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  PATCH_COMMAND patch < "${CMAKE_CURRENT_SOURCE_DIR}/boost.patch"
  )
FetchContent_MakeAvailable(boost)
