set(src pybind.cpp)

pybind11_add_module(_spdl_ffmpeg6 ${src})

target_compile_definitions(_spdl_ffmpeg6 PRIVATE "-DSPDL_FFMPEG_EXT_NAME=_spdl_ffmpeg4")
target_link_libraries(_spdl_ffmpeg6 PRIVATE spdl)
