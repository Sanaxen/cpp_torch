#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TorchVision::TorchVision" for configuration "Release"
set_property(TARGET TorchVision::TorchVision APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TorchVision::TorchVision PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/torchvision.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/torchvision.dll"
  )

list(APPEND _IMPORT_CHECK_TARGETS TorchVision::TorchVision )
list(APPEND _IMPORT_CHECK_FILES_FOR_TorchVision::TorchVision "${_IMPORT_PREFIX}/lib/torchvision.lib" "${_IMPORT_PREFIX}/bin/torchvision.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
