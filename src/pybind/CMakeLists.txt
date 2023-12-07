set(src pybind.cpp)

function(add_spdl_extension ffmpeg_version)
  set(name "_spdl_ffmpeg${ffmpeg_version}")

  pybind11_add_module(${name} ${src})
  target_compile_definitions(${name} PRIVATE "-DSPDL_FFMPEG_EXT_NAME=${name}")
  target_link_libraries(${name} PRIVATE "libspdl_ffmpeg${ffmpeg_version}")

  install(
    TARGETS ${name}
    LIBRARY DESTINATION spdl/lib
    RUNTIME DESTINATION spdl/lib
  )
endfunction()

add_spdl_extension(4)
add_spdl_extension(5)
add_spdl_extension(6)
