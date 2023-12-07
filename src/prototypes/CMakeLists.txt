add_executable(coro_generator coro_generator.cpp)
target_link_libraries(coro_generator folly)

add_executable(spdl_proto proto.cpp)
target_link_libraries(spdl_proto libspdl_ffmpeg4)
target_include_directories(spdl_proto PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")

install(
  TARGETS spdl_proto coro_generator)
