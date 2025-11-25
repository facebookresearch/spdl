# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

message(STATUS "########################################")
message(STATUS "# Configuring libspdl tests")
message(STATUS "########################################")

# Find GTest package


# Generate test data directory
set(TEST_DATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/test_data")
file(MAKE_DIRECTORY "${TEST_DATA_DIR}")

# Helper function to generate test data files using FFmpeg
function(generate_test_data)
  find_program(FFMPEG_EXECUTABLE ffmpeg REQUIRED)

  # Generate test audio (AAC, 5 seconds, 44.1kHz, stereo)
  execute_process(
    COMMAND ${FFMPEG_EXECUTABLE}
      -f lavfi
      -i sine=frequency=440:duration=5:sample_rate=44100
      -ac 2
      -c:a aac
      -b:a 128k
      ${TEST_DATA_DIR}/test_audio.aac
      -y
    OUTPUT_QUIET
    ERROR_QUIET
    RESULT_VARIABLE AUDIO_RESULT
  )

  # Generate test image (JPEG, 320x240)
  execute_process(
    COMMAND ${FFMPEG_EXECUTABLE}
      -f lavfi
      -i testsrc=duration=1:size=320x240:rate=1
      -frames:v 1
      ${TEST_DATA_DIR}/test_image.jpg
      -y
    OUTPUT_QUIET
    ERROR_QUIET
    RESULT_VARIABLE IMAGE_RESULT
  )

  # Generate test video (H.264/MP4, 2 seconds, 320x240, 30fps)
  execute_process(
    COMMAND ${FFMPEG_EXECUTABLE}
      -f lavfi
      -i testsrc=duration=2:size=320x240:rate=30
      -c:v libx264
      -pix_fmt yuv420p
      ${TEST_DATA_DIR}/test_video.mp4
      -y
    OUTPUT_QUIET
    ERROR_QUIET
    RESULT_VARIABLE VIDEO_RESULT
  )

  if(AUDIO_RESULT EQUAL 0 AND IMAGE_RESULT EQUAL 0 AND VIDEO_RESULT EQUAL 0)
    message(STATUS "Test data generated successfully in ${TEST_DATA_DIR}")
  else()
    message(WARNING "Failed to generate some test data files. Tests may fail.")
  endif()
endfunction()

# Generate test data at configure time
generate_test_data()

# Create smoke test executable for each FFmpeg version
function(add_smoke_test ffmpeg_version)
  set(test_name "smoke_test_ffmpeg${ffmpeg_version}")
  set(lib_name "spdl_ffmpeg${ffmpeg_version}")

  message(STATUS "Building test: ${test_name}")

  add_executable(${test_name} smoke_test.cpp)

  target_compile_definitions(${test_name} PRIVATE
    "TEST_DATA_DIR=\"${TEST_DATA_DIR}\""
  )

  target_link_libraries(${test_name}
    PRIVATE
    ${lib_name}
    fmt::fmt
    glog::glog
    GTest::gtest
    GTest::gtest_main
  )

  target_include_directories(${test_name}
    PRIVATE
    "${PROJECT_SOURCE_DIR}"
  )

  # Register test with CTest
  add_test(
    NAME ${test_name}
    COMMAND ${test_name}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  # Set test environment
  set_tests_properties(${test_name} PROPERTIES
    ENVIRONMENT "TEST_DATA_DIR=${TEST_DATA_DIR}"
  )
endfunction()

# Build tests for configured FFmpeg versions
set(ffmpeg_versions 8 7 6 5 4)
if (SPDL_USE_FFMPEG_VERSION IN_LIST ffmpeg_versions)
  add_smoke_test("${SPDL_USE_FFMPEG_VERSION}")
else()
  add_smoke_test(8)
  add_smoke_test(7)
  add_smoke_test(6)
  add_smoke_test(5)
  add_smoke_test(4)
endif()
