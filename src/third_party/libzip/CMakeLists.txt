message(STATUS "########################################")
message(STATUS "# Configuring libzip")
message(STATUS "########################################")

set(ENABLE_COMMONCRYPTO OFF)
set(ENABLE_GNUTLS OFF)
set(ENABLE_MBEDTLS OFF)
set(ENABLE_OPENSSL OFF)
set(ENABLE_WINDOWS_CRYPTO OFF)
set(ENABLE_BZIP2 OFF)
set(ENABLE_LZMA OFF)
set(ENABLE_ZSTD OFF)
set(ENABLE_FDOPEN OFF)
set(BUILD_TOOLS OFF)
set(BUILD_REGRESS OFF)
set(BUILD_OSSFUZZ OFF)
set(BUILD_EXAMPLES OFF)
set(BUILD_DOC OFF)

set(BUILD_SHARED_LIBS OFF)
set(LIBZIP_DO_INSTALL ON)
FetchContent_Declare(
  libzip
  URL https://github.com/nih-at/libzip/releases/download/v1.11.2/libzip-1.11.2.tar.gz
  URL_HASH SHA256=6b2a43837005e1c23fdfee532b78f806863e412d2089b9c42b49ab08cbcd7665
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
FetchContent_MakeAvailable(libzip)
