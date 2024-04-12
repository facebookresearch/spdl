message(STATUS "########################################")
message(STATUS "# Configuring NVDEC (stub)")
message(STATUS "########################################")

FetchContent_Declare(
    cuda_video_stub
    URL https://github.com/mthrok/cuda_video_stub/archive/refs/tags/0.1.0.tar.gz
    URL_HASH SHA256=7d66efe4db7b30aefc051507f868c9469b281ed0061acc9a53e7151ebbde6f88
    DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )
FetchContent_MakeAvailable(cuda_video_stub)
