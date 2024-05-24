message(STATUS "########################################")
message(STATUS "# Configuring nvJPEG (stub)")
message(STATUS "########################################")

FetchContent_Declare(
    nvjpeg_stub
    URL https://github.com/mthrok/nvjpeg_stub/archive/refs/tags/v0.1.0.tar.gz
    URL_HASH SHA256=39d51d35f796892c5e5cc9572f7b307418f9818ae191113796db6774c813926b
    DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
    )
FetchContent_MakeAvailable(nvjpeg_stub)
