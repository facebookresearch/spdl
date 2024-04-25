set(src
  bind.cpp
  core/adaptors.cpp
  core/conversion.cpp
  core/decoding.cpp
  core/demuxing.cpp
  core/executor.cpp
  core/frames_and_buffers.cpp
  core/future.cpp
  core/packets.cpp
  core/tracing.cpp
  core/utils.cpp)

# This adds the library directory to search path
if (APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
elseif (UNIX)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
endif ()


if (NOT DEFINED SPDL_PYTHON_BINDING_INSTALL_PREFIX)
  set(SPDL_PYTHON_BINDING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif ()

function(add_spdl_extension ffmpeg_version)
  set(name "_spdl_ffmpeg${ffmpeg_version}")

  set(defs "-DSPDL_FFMPEG_EXT_NAME=${name}")
  if (SPDL_USE_CUDA)
    list(APPEND defs SPDL_USE_CUDA)
  endif()
  if (SPDL_USE_NVCODEC)
    list(APPEND defs SPDL_USE_NVCODEC)
  endif ()

  nanobind_add_module(${name} ${src})
  target_compile_definitions(${name} PRIVATE "${defs}")
  target_link_libraries(${name} PRIVATE "spdl_ffmpeg${ffmpeg_version}")
  target_include_directories(${name} PRIVATE "${Python_INCLUDE_DIR}")
  target_include_directories(nanobind-static PRIVATE "${Python_INCLUDE_DIR}")

  install(
    TARGETS ${name}
    LIBRARY DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
    RUNTIME DESTINATION "${SPDL_PYTHON_BINDING_INSTALL_PREFIX}/lib"
  )
endfunction()

add_spdl_extension(4)
add_spdl_extension(5)
add_spdl_extension(6)
add_spdl_extension(7)
