message(STATUS "########################################")
message(STATUS "# Configuring Nvidia Viceo Codec (stub)")
message(STATUS "########################################")

FetchContent_Declare(
    nvcodec_stub
    URL https://github.com/mthrok/nvcodec_stub/archive/refs/tags/0.1.0.tar.gz
    URL_HASH SHA256=6a26b5605456a374ec424f3c4f30c99247afdc299991ab5f6142e39132394894
    DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )
FetchContent_MakeAvailable(nvcodec_stub)
