message(STATUS "########################################")
message(STATUS "# Configuring Nvidia Viceo Codec (stub)")
message(STATUS "########################################")

FetchContent_Declare(
    nvcodec_stub
    URL https://github.com/mthrok/nvcodec_stub/archive/refs/tags/0.2.0.tar.gz
    URL_HASH SHA256=e811abfc07062793b7e578092c069e4dcab4d315ff9e748cfebbea7ea15b2b48
    DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )
FetchContent_MakeAvailable(nvcodec_stub)
