# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Keep this dependency out of ordinary library/wheel builds.
get_target_property(glog_type glog::glog TYPE)
if (NOT glog_type STREQUAL "SHARED_LIBRARY")
  message(FATAL_ERROR
    "Native tests need one shared glog instance for warning capture. "
    "Build src/third_party with -DSPDL_BUILD_TESTS=ON, or use tools/build_cpp_tests.sh.")
endif()

set(BUILD_GMOCK OFF)
set(INSTALL_GTEST OFF)
if (MSVC)
  set(gtest_force_shared_crt ON)
endif()
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
  URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
  DOWNLOAD_DIR "${PROJECT_SOURCE_DIR}/third_party/cache"
  )
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# Generate the same kinds of fixtures as the Buck tests, without Buck resources
# or network access during testing. The CLI need not match the tested FFmpeg ABI.
find_program(SPDL_TEST_FFMPEG_EXECUTABLE NAMES ffmpeg REQUIRED)
# Prefer the full test environment for transitive FFmpeg runtime dependencies.
set(test_data_dir "${CMAKE_CURRENT_BINARY_DIR}/data")
set(test_data
  "${test_data_dir}/test_audio.aac"
  "${test_data_dir}/test_image.jpg"
  "${test_data_dir}/test_video.mp4"
  )
add_custom_command(
  OUTPUT ${test_data}
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${test_data_dir}"
  COMMAND "${SPDL_TEST_FFMPEG_EXECUTABLE}" -nostdin -hide_banner -loglevel error -y
    -f lavfi -i "sine=frequency=440:duration=5:sample_rate=44100"
    -ac 2 -c:a aac -b:a 128k "${test_data_dir}/test_audio.aac"
  COMMAND "${SPDL_TEST_FFMPEG_EXECUTABLE}" -nostdin -hide_banner -loglevel error -y
    -f lavfi -i "testsrc=duration=1:size=320x240:rate=1"
    -frames:v 1 "${test_data_dir}/test_image.jpg"
  COMMAND "${SPDL_TEST_FFMPEG_EXECUTABLE}" -nostdin -hide_banner -loglevel error -y
    -f lavfi -i "testsrc=duration=2:size=320x240:rate=30"
    -c:v libx264 -pix_fmt yuv420p "${test_data_dir}/test_video.mp4"
  COMMENT "Generating libspdl test media"
  VERBATIM
  )
add_custom_target(libspdl_test_data DEPENDS ${test_data})

set(test_sources
  buffer_test.cpp
  bytes_test.cpp
  codec_test.cpp
  encoder_test.cpp
  frames_test.cpp
  muxer_test.cpp
  packets_serialization_test.cpp
  packets_test.cpp
  rational_utils_test.cpp
  smoke_test.cpp
  storage_test.cpp
  wrappers_test.cpp
  )

add_custom_target(libspdl_tests)
foreach(version 9 8 7 6 5 4)
  if (TARGET "spdl_ffmpeg${version}")
    set(target "libspdl_core_test_ffmpeg${version}")
    add_executable("${target}" ${test_sources})
    target_link_libraries("${target}" PRIVATE
      "spdl_ffmpeg${version}" "ffmpeg${version}"
      GTest::gtest_main fmt::fmt glog::glog
      )
    target_compile_definitions("${target}" PRIVATE
      "TEST_DATA_DIR=\"${test_data_dir}\""
      )
    add_dependencies("${target}" libspdl_test_data)
    if (SPDL_TEST_FFMPEG_ROOT)
      set_property(TARGET "${target}" APPEND PROPERTY BUILD_RPATH
        "${SPDL_TEST_FFMPEG_ROOT}/lib")
      set_property(TARGET "spdl_ffmpeg${version}" APPEND PROPERTY BUILD_RPATH
        "${SPDL_TEST_FFMPEG_ROOT}/lib")
    endif()
    add_dependencies(libspdl_tests "${target}")
    gtest_discover_tests("${target}"
      TEST_PREFIX "ffmpeg${version}."
      DISCOVERY_MODE PRE_TEST
      DISCOVERY_TIMEOUT 30
      PROPERTIES TIMEOUT 120 LABELS "ffmpeg${version}"
      )
  endif()
endforeach()
