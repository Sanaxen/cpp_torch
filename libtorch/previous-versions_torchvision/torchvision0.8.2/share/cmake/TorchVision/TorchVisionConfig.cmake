# TorchVisionConfig.cmake
# --------------------
#
# Exported targets:: Vision
#


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was TorchVisionConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(PN TorchVision)

# location of include/torchvision
set(${PN}_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/include")

set(${PN}_LIBRARY "")
set(${PN}_DEFINITIONS USING_${PN})

check_required_components(${PN})


if(NOT (CMAKE_VERSION VERSION_LESS 3.0))
#-----------------------------------------------------------------------------
# Don't include targets if this file is being picked up by another
# project which has already built this as a subproject
#-----------------------------------------------------------------------------
if(NOT TARGET ${PN}::TorchVision)
include("${CMAKE_CURRENT_LIST_DIR}/${PN}Targets.cmake")

if(NOT TARGET torch_library)
find_package(Torch REQUIRED)
endif()
if(NOT TARGET Python3::Python)
find_package(Python3 COMPONENTS Development)
endif()

set_target_properties(TorchVision::TorchVision PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PACKAGE_PREFIX_DIR}/include" INTERFACE_LINK_LIBRARIES "torch;Python3::Python" )


if(OFF)
  target_compile_definitions(TorchVision::TorchVision INTERFACE WITH_CUDA)
endif()

endif()
endif()
