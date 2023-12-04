message(STATUS "########################################")
message(STATUS "Configuring Boost")
message(STATUS "########################################")

FetchContent_Declare(
  boost
  GIT_REPOSITORY https://github.com/boostorg/boost.git
  GIT_TAG boost-1.83.0
  GIT_SUBMODULES
  tools/cmake
  libs/algorithm
  libs/array
  libs/assert
  libs/bind
  libs/exception
  libs/config
  libs/concept_check
  libs/conversion
  libs/container_hash
  libs/context
  libs/core
  libs/describe
  libs/detail
  libs/function
  libs/function_types
  libs/functional
  libs/fusion
  libs/integer
  libs/intrusive
  libs/io
  libs/iterator
  libs/move
  libs/mp11
  libs/mpl
  libs/optional
  libs/pool
  libs/predef
  libs/preprocessor
  libs/range
  libs/regex
  libs/unordered
  libs/static_assert
  libs/smart_ptr
  libs/tuple
  libs/type_index
  libs/typeof
  libs/utility
  libs/variant
  libs/winapi
  libs/throw_exception
  libs/type_traits
  )
FetchContent_MakeAvailable(boost)
