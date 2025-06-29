###############################################################################
# Global config
###############################################################################
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
message(STATUS "CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

cmake_policy(SET CMP0042 NEW) # MACOSX_RPATH is enabled by default.
cmake_policy(SET CMP0068 NEW) # RPATH settings on macOS do not affect install_name.

###############################################################################
# CCache
###############################################################################
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Found ccache")
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
  set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
else()
  message(STATUS "Could not find ccache. Consider installing ccache to speed up compilation.")
endif()

###############################################################################
# FetchContent config
###############################################################################
include(FetchContent)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

# somehow `cmake_policy(SET .. NEW)` does not work for those...
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)


###############################################################################
# Helper functions
###############################################################################

# https://stackoverflow.com/questions/32183975/how-to-print-all-the-properties-of-a-target-in-cmake
if(NOT CMAKE_PROPERTY_LIST)
    execute_process(COMMAND cmake --help-property-list OUTPUT_VARIABLE CMAKE_PROPERTY_LIST)
    # Convert command output into a CMake list
    string(REGEX REPLACE ";" "\\\\;" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    string(REGEX REPLACE "\n" ";" CMAKE_PROPERTY_LIST "${CMAKE_PROPERTY_LIST}")
    list(REMOVE_DUPLICATES CMAKE_PROPERTY_LIST)
endif()

function(print_target_properties target)
    if(NOT TARGET ${target})
      message(STATUS "There is no target named '${target}'")
      return()
    endif()

    foreach(property ${CMAKE_PROPERTY_LIST})
        string(REPLACE "<CONFIG>" "${CMAKE_BUILD_TYPE}" property ${property})

        # Fix https://stackoverflow.com/questions/32197663/how-can-i-remove-the-the-location-property-may-not-be-read-from-target-error-i
        if(property STREQUAL "LOCATION" OR property MATCHES "^LOCATION_" OR property MATCHES "_LOCATION$")
            continue()
        endif()

        get_property(was_set TARGET ${target} PROPERTY ${property} SET)
        if(was_set)
            get_target_property(value ${target} ${property})
            message(STATUS "${target} ${property} = ${value}")
        endif()
    endforeach()
endfunction()

set(
  SPDL_CXX_CPU_COMPILE_FLAGS
  "-Werror=deprecated"
  "-Werror=exceptions"
  "-Wextra"
  "-Wextra-semi"
  "-Werror=missing-field-initializers"
  "-Wmissing-noreturn"
  "-Werror=mismatched-tags"
  "-Wshadow"
  "-Wunused-function"
  "-Werror=unused-value"
  "-Werror=unused-variable"
  "-Wzero-as-null-pointer-constant"
  )
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  list(APPEND
    SPDL_CXX_CPU_COMPILE_FLAGS
    "-Werror=ambiguous-reversed-operator"
    "-Werror=deprecated-this-capture"
    "-Werror=deprecated-copy-with-dtor"
    "-Werror=deprecated-copy-with-user-provided-copy"
    "-Werror=deprecated-copy-with-user-provided-dtor"
    "-Werror=deprecated-dynamic-exception-spec"
    "-Werror=duplicate-enum"
    "-Wextra-semi-stmt"
    "-Werror=implicit-const-int-float-conversion"
    "-Wmissing-prototypes"
    "-Werror=shorten-64-to-32"
    "-Werror=reorder-init-list"
    # -Wunsafe-buffer-usage
    "-Werror=unused-exception-parameter"
    "-Werror=unused-lambda-capture"
    "-Wunused-private-field"
    )
endif()
