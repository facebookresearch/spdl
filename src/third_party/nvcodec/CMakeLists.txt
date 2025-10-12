message(STATUS "########################################")
message(STATUS "# Configuring Nvidia Viceo Codec (stub)")
message(STATUS "########################################")

FetchContent_Declare(
    nvcodec_stub
    URL https://github.com/mthrok/nvcodec_stub/archive/refs/tags/0.3.0.tar.gz
    URL_HASH SHA256=249bd10d0040b1ed30ff7813e735760c121f3b9040eeb01106238d30c3e577fc
    DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )
FetchContent_MakeAvailable(nvcodec_stub)
