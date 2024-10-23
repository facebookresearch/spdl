message(STATUS "########################################")
message(STATUS "# Configuring robin-map")
message(STATUS "########################################")

FetchContent_Declare(
  robin-map
  URL https://github.com/Tessil/robin-map/archive/refs/tags/v1.3.0.tar.gz
  URL_HASH SHA256=a8424ad3b0affd4c57ed26f0f3d8a29604f0e1f2ef2089f497f614b1c94c7236
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
FetchContent_MakeAvailable(robin-map)

message(STATUS "########################################")
message(STATUS "# Configuring nanobind")
message(STATUS "########################################")

FetchContent_Declare(
  nanobind
  URL https://github.com/wjakob/nanobind/archive/refs/tags/v2.2.0.zip
  URL_HASH SHA256=f7f96f730faf78a9d88761f7fd883f2ba7d851d33be3d9a103a1bdbb270cce14
  DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  )
if(NOT nanobind_POPULATED)
  FetchContent_Populate(nanobind)
  file(COPY "${robin-map_SOURCE_DIR}/include" DESTINATION "${nanobind_SOURCE_DIR}/ext/robin_map/")
  add_subdirectory(${nanobind_SOURCE_DIR} ${nanobind_BINARY_DIR})
endif()
