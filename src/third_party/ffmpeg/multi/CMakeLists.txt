message(STATUS "########################################")
message(STATUS "# Configuring FFmpeg")
message(STATUS "########################################")

################################################################################
# This file defines the following FFmpeg libraries using pre-built binaries.

add_library(ffmpeg4 INTERFACE)
add_library(ffmpeg5 INTERFACE)
add_library(ffmpeg6 INTERFACE)
add_library(ffmpeg7 INTERFACE)
add_library(ffmpeg8 INTERFACE)

################################################################################

set(base_url https://github.com/facebookresearch/spdl/releases/download/release-assets/)

if (APPLE)
  if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    FetchContent_Declare(
      f4
      URL ${base_url}/ffmpeg_4.4.5_macos_aarch64.tar.gz
      URL_HASH SHA256=703ee367571cf8f64f5f0535cbfa1f3b2889fde64461717a55bcce45f0a41ca0
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f5
      URL ${base_url}/ffmpeg_5.1.6_macos_aarch64.tar.gz
      URL_HASH SHA256=d5a4f24b948462287f9198c5fc5d3ea0e85114ad0d13089799a76b57a96c9d03
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f6
      URL ${base_url}/ffmpeg_6.1.2_macos_aarch64.tar.gz
      URL_HASH SHA256=581100ff990a053de84361cf6517ec880674f159196ba208beec70178fd29807
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f7
      URL ${base_url}/ffmpeg_7.0.2_macos_aarch64.tar.gz
      URL_HASH SHA256=29bcf0c4b8c7dc9c28c6dc18b59c43be302ad5af9c4ee77e8c0c76072531485f
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f8
      URL ${base_url}/ffmpeg_8.0_macos_aarch64.tar.gz
      URL_HASH SHA256=c33d6df6aa2bce5152a1db3c6accdfc16fefdf4cfe61dcaf671cee6d4256037f
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
  else ()
    message(
      FATAL_ERROR
      "${CMAKE_SYSTEM_PROCESSOR} is not supported for FFmpeg multi-version integration. "
      "If you have FFmpeg libraries installed in the system,"
      " setting FFMPEG_ROOT environment variable to the root directory of FFmpeg installation"
      " (the directory where `include` and `lib` sub directories with corresponding headers"
      " and library files are present) will invoke the FFmpeg single-version integration. "
      "If you do not need the FFmpeg integration, setting USE_FFMPEG=0 will bypass the issue.")
  endif()
elseif (UNIX)
  if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "aarch64")
    FetchContent_Declare(
      f4
      URL ${base_url}/ffmpeg_4.4.5_manylinux_2_28_aarch64.tar.gz
      URL_HASH SHA256=0f07b8e76be3d03b8e074d3da7a07c2030ca62137fe78282424d428d23efdf09
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f5
      URL ${base_url}/ffmpeg_5.1.6_manylinux_2_28_aarch64.tar.gz
      URL_HASH SHA256=f00a05aae720ce1c9eb40450c4f8f45436c47807efe6cd04b37a9cfb3df0ad5a
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f6
      URL ${base_url}/ffmpeg_6.1.2_manylinux_2_28_aarch64.tar.gz
      URL_HASH SHA256=99f2bea858c189c76e63fb937069569b24acd9f8fc2ee00ea6eeb6a0a449bcb6
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f7
      URL ${base_url}/ffmpeg_7.0.2_manylinux_2_28_aarch64.tar.gz
      URL_HASH SHA256=1cc22ec6607c06ff9578affe4a502eb60e86c49309606e192b297b6c49c35b48
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f8
      URL ${base_url}/ffmpeg_8.0_manylinux_2_28_aarch64.tar.gz
      URL_HASH SHA256=45990fa09bfb32fbba28644677efe06ffe48d23d6913acb6911967367bc42e91
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
  elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    FetchContent_Declare(
      f4
      URL ${base_url}/ffmpeg_4.4.5_manylinux_2_28_x86_64.tar.gz
      URL_HASH SHA256=f2aefe3ef4dd8841b2092c2b50c6342444da1b13d1dc9dc722b23c0632013457
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f5
      URL ${base_url}/ffmpeg_5.1.6_manylinux_2_28_x86_64.tar.gz
      URL_HASH SHA256=f94aa7a59b0b6bc78b5a92a2298147d89f748c9d1335c97b248f09aae3088757
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f6
      URL ${base_url}/ffmpeg_6.1.2_manylinux_2_28_x86_64.tar.gz
      URL_HASH SHA256=91a126132d71665a16cc63e74746e3a618deae5220340bc57bfb7befbceaf9b7
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f7
      URL ${base_url}/ffmpeg_7.0.2_manylinux_2_28_x86_64.tar.gz
      URL_HASH SHA256=a30e9a3846cc322e95c483788bec35591c3e4398390db9bc833a4bc15662ebad
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f8
      URL ${base_url}/ffmpeg_8.0_manylinux_2_28_x86_64.tar.gz
      URL_HASH SHA256=7be341ae94e5dcdd17a419f237365db03aaea6aa9550fb745469ffdde15b98b1
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
  else ()
    # Possible case ppc64le (though it's not officially supported.)
    message(
      FATAL_ERROR
      "${CMAKE_SYSTEM_PROCESSOR} is not supported for FFmpeg multi-version integration. "
      "If you have FFmpeg libraries installed in the system,"
      " setting FFMPEG_ROOT environment variable to the root directory of FFmpeg installation"
      " (the directory where `include` and `lib` sub directories with corresponding headers"
      " and library files are present) will invoke the FFmpeg single-version integration. "
      "If you do not need the FFmpeg integration, setting USE_FFMPEG=0 will bypass the issue.")
  endif()
elseif(MSVC)
  message(
    FATAL_ERROR
    "Windows is not supported.")
else()
  message(FATAL_ERROR "Unsupported platform.")
endif()

FetchContent_MakeAvailable(f4 f5 f6 f7 f8)
target_include_directories(ffmpeg4 INTERFACE ${f4_SOURCE_DIR}/include)
target_include_directories(ffmpeg5 INTERFACE ${f5_SOURCE_DIR}/include)
target_include_directories(ffmpeg6 INTERFACE ${f6_SOURCE_DIR}/include)
target_include_directories(ffmpeg7 INTERFACE ${f7_SOURCE_DIR}/include)
target_include_directories(ffmpeg8 INTERFACE ${f8_SOURCE_DIR}/include)

if(APPLE)
  target_link_libraries(
    ffmpeg4
    INTERFACE
    ${f4_SOURCE_DIR}/lib/libavutil.56.dylib
    ${f4_SOURCE_DIR}/lib/libavcodec.58.dylib
    ${f4_SOURCE_DIR}/lib/libavformat.58.dylib
    ${f4_SOURCE_DIR}/lib/libavdevice.58.dylib
    ${f4_SOURCE_DIR}/lib/libavfilter.7.dylib
    )
  target_link_libraries(
    ffmpeg5
    INTERFACE
    ${f5_SOURCE_DIR}/lib/libavutil.57.dylib
    ${f5_SOURCE_DIR}/lib/libavcodec.59.dylib
    ${f5_SOURCE_DIR}/lib/libavformat.59.dylib
    ${f5_SOURCE_DIR}/lib/libavdevice.59.dylib
    ${f5_SOURCE_DIR}/lib/libavfilter.8.dylib
    )
  target_link_libraries(
    ffmpeg6
    INTERFACE
    ${f6_SOURCE_DIR}/lib/libavutil.58.dylib
    ${f6_SOURCE_DIR}/lib/libavcodec.60.dylib
    ${f6_SOURCE_DIR}/lib/libavformat.60.dylib
    ${f6_SOURCE_DIR}/lib/libavdevice.60.dylib
    ${f6_SOURCE_DIR}/lib/libavfilter.9.dylib
    )
  target_link_libraries(
    ffmpeg7
    INTERFACE
    ${f7_SOURCE_DIR}/lib/libavutil.59.dylib
    ${f7_SOURCE_DIR}/lib/libavcodec.61.dylib
    ${f7_SOURCE_DIR}/lib/libavformat.61.dylib
    ${f7_SOURCE_DIR}/lib/libavdevice.61.dylib
    ${f7_SOURCE_DIR}/lib/libavfilter.10.dylib
    )
  target_link_libraries(
    ffmpeg8
    INTERFACE
    ${f8_SOURCE_DIR}/lib/libavutil.60.dylib
    ${f8_SOURCE_DIR}/lib/libavcodec.62.dylib
    ${f8_SOURCE_DIR}/lib/libavformat.62.dylib
    ${f8_SOURCE_DIR}/lib/libavdevice.62.dylib
    ${f8_SOURCE_DIR}/lib/libavfilter.11.dylib
    )
elseif (UNIX)
  target_link_libraries(
    ffmpeg4
    INTERFACE
    ${f4_SOURCE_DIR}/lib/libavutil.so.56
    ${f4_SOURCE_DIR}/lib/libavcodec.so.58
    ${f4_SOURCE_DIR}/lib/libavformat.so.58
    ${f4_SOURCE_DIR}/lib/libavdevice.so.58
    ${f4_SOURCE_DIR}/lib/libavfilter.so.7
    )
  target_link_libraries(
    ffmpeg5
    INTERFACE
    ${f5_SOURCE_DIR}/lib/libavutil.so.57
    ${f5_SOURCE_DIR}/lib/libavcodec.so.59
    ${f5_SOURCE_DIR}/lib/libavformat.so.59
    ${f5_SOURCE_DIR}/lib/libavdevice.so.59
    ${f5_SOURCE_DIR}/lib/libavfilter.so.8
    )
  target_link_libraries(
    ffmpeg6
    INTERFACE
    ${f6_SOURCE_DIR}/lib/libavutil.so.58
    ${f6_SOURCE_DIR}/lib/libavcodec.so.60
    ${f6_SOURCE_DIR}/lib/libavformat.so.60
    ${f6_SOURCE_DIR}/lib/libavdevice.so.60
    ${f6_SOURCE_DIR}/lib/libavfilter.so.9
    )
  target_link_libraries(
    ffmpeg7
    INTERFACE
    ${f7_SOURCE_DIR}/lib/libavutil.so.59
    ${f7_SOURCE_DIR}/lib/libavcodec.so.61
    ${f7_SOURCE_DIR}/lib/libavformat.so.61
    ${f7_SOURCE_DIR}/lib/libavdevice.so.61
    ${f7_SOURCE_DIR}/lib/libavfilter.so.10
    )
  target_link_libraries(
    ffmpeg8
    INTERFACE
    ${f8_SOURCE_DIR}/lib/libavutil.so.60
    ${f8_SOURCE_DIR}/lib/libavcodec.so.62
    ${f8_SOURCE_DIR}/lib/libavformat.so.62
    ${f8_SOURCE_DIR}/lib/libavdevice.so.62
    ${f8_SOURCE_DIR}/lib/libavfilter.so.11
    )
elseif(MSVC)
  target_link_libraries(
    ffmpeg4
    INTERFACE
    ${f4_SOURCE_DIR}/bin/avutil.lib
    ${f4_SOURCE_DIR}/bin/avcodec.lib
    ${f4_SOURCE_DIR}/bin/avformat.lib
    ${f4_SOURCE_DIR}/bin/avdevice.lib
    ${f4_SOURCE_DIR}/bin/avfilter.lib
    )
  target_link_libraries(
    ffmpeg5
    INTERFACE
    ${f5_SOURCE_DIR}/bin/avutil.lib
    ${f5_SOURCE_DIR}/bin/avcodec.lib
    ${f5_SOURCE_DIR}/bin/avformat.lib
    ${f5_SOURCE_DIR}/bin/avdevice.lib
    ${f5_SOURCE_DIR}/bin/avfilter.lib
    )
  target_link_libraries(
    ffmpeg6
    INTERFACE
    ${f6_SOURCE_DIR}/bin/avutil.lib
    ${f6_SOURCE_DIR}/bin/avcodec.lib
    ${f6_SOURCE_DIR}/bin/avformat.lib
    ${f6_SOURCE_DIR}/bin/avdevice.lib
    ${f6_SOURCE_DIR}/bin/avfilter.lib
    )
  target_link_libraries(
    ffmpeg7
    INTERFACE
    ${f7_SOURCE_DIR}/bin/avutil.lib
    ${f7_SOURCE_DIR}/bin/avcodec.lib
    ${f7_SOURCE_DIR}/bin/avformat.lib
    ${f7_SOURCE_DIR}/bin/avdevice.lib
    ${f7_SOURCE_DIR}/bin/avfilter.lib
    )
  target_link_libraries(
    ffmpeg8
    INTERFACE
    ${f8_SOURCE_DIR}/bin/avutil.lib
    ${f8_SOURCE_DIR}/bin/avcodec.lib
    ${f8_SOURCE_DIR}/bin/avformat.lib
    ${f8_SOURCE_DIR}/bin/avdevice.lib
    ${f8_SOURCE_DIR}/bin/avfilter.lib
    )
endif()
