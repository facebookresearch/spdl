message(STATUS "########################################")
message(STATUS "# Configuring Nvidia Viceo Codec (stub)")
message(STATUS "########################################")

FetchContent_Declare(
    nvcodec_stub
    URL https://github.com/mthrok/nvcodec_stub/archive/refs/tags/0.1.0.tar.gz
    URL_HASH SHA256=7d66efe4db7b30aefc051507f868c9469b281ed0061acc9a53e7151ebbde6f88
    DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )
FetchContent_MakeAvailable(nvcodec_stub)
