set(src pybind.cpp)

# This adds the library directory to search path
if (APPLE)
  set(CMAKE_INSTALL_RPATH "@loader_path")
elseif (UNIX)
  set(CMAKE_INSTALL_RPATH $ORIGIN)
endif ()

function(add_spdl_extension ffmpeg_version)
  set(name "_spdl_ffmpeg${ffmpeg_version}")

  pybind11_add_module(${name} ${src})
  target_compile_definitions(${name} PRIVATE "-DSPDL_FFMPEG_EXT_NAME=${name}")
  target_link_libraries(${name} PRIVATE "spdl_ffmpeg${ffmpeg_version}_static")

  install(
    TARGETS ${name}
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION lib
  )
endfunction()

add_spdl_extension(4)
add_spdl_extension(5)
add_spdl_extension(6)
