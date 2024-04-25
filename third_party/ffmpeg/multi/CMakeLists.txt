message(STATUS "########################################")
message(STATUS "# Configuring FFmpeg")
message(STATUS "########################################")

################################################################################
# This file defines the following FFmpeg libraries using pre-built binaries.

add_library(ffmpeg4 INTERFACE)
add_library(ffmpeg5 INTERFACE)
add_library(ffmpeg6 INTERFACE)
add_library(ffmpeg7 INTERFACE)

################################################################################

set(base_url https://pytorch.s3.amazonaws.com/torchaudio/ffmpeg)

if (APPLE)
  if ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "arm64")
    FetchContent_Declare(
      f4
      URL ${base_url}/2023-08-14/macos_arm64/4.4.4.tar.gz
      URL_HASH SHA256=9a593eb241eb8b23bc557856ee6db5d9aecd2d8895c614a949f3a1ad9799c1a1
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f5
      URL ${base_url}/2023-07-06/macos_arm64/5.0.3.tar.gz
      URL_HASH SHA256=316fe8378afadcf63089acf3ad53a626fd3c26cc558b96ce1dc94d2a78f4deb4
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f6
      URL ${base_url}/2023-07-06/macos_arm64/6.0.tar.gz
      URL_HASH SHA256=5d1da9626f8cb817d6c558a2c61085a3d39a8d9f725a6f69f4658bea8efa9389
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f7
      URL ${base_url}/2024-04-05/macos_arm64/7.0.tar.gz
      URL_HASH SHA256=f73dc42e96eb59ac34689f59e59cd140f0c865574aee5b9f7ded9ef16b6738e8
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
  elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    FetchContent_Declare(
      f4
      URL ${base_url}/2023-08-14/macos_x86_64/4.4.4.tar.gz
      URL_HASH SHA256=0935e359c0864969987d908397f9208d6dc4dc0ef8bfe2ec730bb2c44eae89fc
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f5
      URL ${base_url}/2023-07-06/macos_x86_64/5.0.3.tar.gz
      URL_HASH SHA256=d0b49575d3b174cfcca53b3049641855e48028cf22dd32f3334bbec4ca94f43e
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f6
      URL ${base_url}/2023-07-06/macos_x86_64/6.0.tar.gz
      URL_HASH SHA256=eabc01eb7d9e714e484d5e1b27bf7d921e87c1f3c00334abd1729e158d6db862
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f7
      URL ${base_url}/2024-04-05/macos_x86_64/7.0.tar.gz
      URL_HASH SHA256=a7ea9df684f90993a213d458427b21b13fb8672b2734637701c1f75282530041
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
      URL ${base_url}/2023-08-14/linux_aarch64/4.4.4.tar.gz
      URL_HASH SHA256=6f00437d13a3b3812ebe81c6e6f3a84a58f260d946a1995df87ba09aae234504
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f5
      URL ${base_url}/2023-07-06/linux_aarch64/5.0.3.tar.gz
      URL_HASH SHA256=65c663206982ee3f0ff88436d8869d191b46061e01e753518c77ecc13ea0236d
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f6
      URL ${base_url}/2023-07-06/linux_aarch64/6.0.tar.gz
      URL_HASH SHA256=ec762fd41ea7b8d9ad4f810f53fd78a565f2bc6f680afe56d555c80f3d35adef
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f7
      URL ${base_url}/2024-04-05/linux_aarch64/7.0.tar.gz
      URL_HASH SHA256=4bb2a2a5b8c547bfcc631073d6f97939666db8f5a85e85d3262ab07608e0fe67
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
  elseif ("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64")
    FetchContent_Declare(
      f4
      URL ${base_url}/2023-08-14/linux_x86_64/4.4.4.tar.gz
      URL_HASH SHA256=9b87eeba9b6975e25f28ba12163bd713228ed84f4c2b3721bc5ebe92055edb51
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f5
      URL ${base_url}/2023-07-06/linux_x86_64/5.0.3.tar.gz
      URL_HASH SHA256=de3c75c99b9ce33de7efdc144566804ae5880457ce71e185b3f592dc452edce7
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f6
      URL ${base_url}/2023-07-06/linux_x86_64/6.0.tar.gz
      URL_HASH SHA256=04d3916404bab5efadd29f68361b7d13ea71e6242c6473edcb747a41a9fb97a6
      DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
      )
    FetchContent_Declare(
      f7
      URL ${base_url}/2024-04-05/linux_x86_64/7.0.tar.gz
      URL_HASH SHA256=4b21ce39cc821c371d9cd1fb64d0f1d7a74279ed10675cf7116ae04b4ad8c58b
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
  FetchContent_Declare(
    f4
    URL ${base_url}/2023-08-14/windows/4.4.4.tar.gz
    URL_HASH SHA256=9f9a65cf03a3e164edca601ba18180a504e44e03fae48ce706ca3120b55a4db5
    DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
  FetchContent_Declare(
    f5
    URL ${base_url}/2023-07-06/windows/5.0.3.tar.gz
    URL_HASH SHA256=e2daa10799909e366cb1b4b91a217d35f6749290dcfeea40ecae3d5b05a46cb3
    DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
  FetchContent_Declare(
    f6
    URL ${base_url}/2023-07-06/windows/6.0.tar.gz
    URL_HASH SHA256=098347eca8cddb5aaa61e9ecc1a00548c645fc59b4f7346b3d91414aa00a9cf6
    DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
  FetchContent_Declare(
    f7
    URL ${base_url}/2024-04-05/windows/7.0.tar.gz
    URL_HASH SHA256=a53658b53d6da34a67dade6edd529beb33b250b8ed906c89832805f2226d430f
    DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
endif()

FetchContent_MakeAvailable(f4 f5 f6 f7)
target_include_directories(ffmpeg4 INTERFACE ${f4_SOURCE_DIR}/include)
target_include_directories(ffmpeg5 INTERFACE ${f5_SOURCE_DIR}/include)
target_include_directories(ffmpeg6 INTERFACE ${f6_SOURCE_DIR}/include)
target_include_directories(ffmpeg7 INTERFACE ${f6_SOURCE_DIR}/include)

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
endif()
