add_executable(coro_generator coro_generator.cpp)
target_link_libraries(coro_generator PRIVATE folly)

add_executable(spdl_proto proto.cpp)
target_link_libraries(spdl_proto spdl_ffmpeg6)
target_include_directories(spdl_proto PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")

install(TARGETS spdl_proto coro_generator)
