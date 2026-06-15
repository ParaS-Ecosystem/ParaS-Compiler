cmake_minimum_required(VERSION 3.26.5)
project(ParaS-v1.0)

message(STATUS "Configuring paras v1.0")

# Set GCC toolchain path for Clang
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# Add the GCC toolchain flags
##set(CMAKE_CXX_FLAGS "--gcc-toolchain=/usr/local/" CACHE STRING "CXX flags" FORCE)
##set(CMAKE_EXE_LINKER_FLAGS "--gcc-toolchain=/usr/local/" CACHE STRING "Linker flags" FORCE)
##set(CMAKE_SHARED_LINKER_FLAGS "--gcc-toolchain=/usr/local/" CACHE STRING "Shared linker flags" FORCE)

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 20)
endif()

#setting paras version info

if(NOT DEFINED PARAS_MAJOR_VERSION)
	set(PARAS_MAJOR_VERSION 1)
endif()

if(NOT DEFINED PARAS_MINOR_VERSION)
	set(PARAS_MINOR_VERSION 0)
endif()

if(NOT DEFINED PARAS_VERSION_PATCH)
	set(PARAS_VERSION_PATCH 0)
endif()

if (NOT PARAS_VERSION)
	set(PARAS_VERSION "${PARAS_MAJOR_VERSION}.${PARAS_MINOR_VERSION}.${PARAS_VERSION_PATCH}")
endif()


message(STATUS "Setting up configuration for PARAS v${PARAS_VERSION}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

#include_directories(${PROJECT_SOURCE_DIR}/include)

if(NOT DEFINED PARAS_INSTALL_PREFIX)
	message(FATAL_ERROR "Install path required. Use -DPARAS_INSTALL_PREFIX to specify")
else()
	cmake_path(ABSOLUTE_PATH PARAS_INSTALL_PREFIX
		BASE_DIRECTORY ${CMAKE_BINARY_DIR}
		OUTPUT_VARIABLE CMAKE_INSTALL_PREFIX)

	message(STATUS "Setting up Paths ... ${CMAKE_INSTALL_PREFIX}")
	if(NOT DEFINED PARAS_BASE_COMPILER)
		message(FATAL_ERROR "ParaS base compiler not specified. Use -DPARAS_BASE_COMPILER to specify")
	else()
		file(WRITE ${CMAKE_INSTALL_PREFIX}/include/pathfinder.hpp
"#ifndef __PARAS_PATHFINDER_HPP__
#define __PARAS_PATHFINDER_HPP__

#define PARAS_INSTALL_PREFIX \"${CMAKE_INSTALL_PREFIX}\"
#define PARAS_BASE_COMPILER \"${PARAS_BASE_COMPILER}\"
#endif
")
	endif()
endif()

add_subdirectory(include)
add_subdirectory(src)

######################## New module ##############################################################

# ================== ParaSCC Install Module ==================

include(CMakePackageConfigHelpers)

set(PARAS_INSTALL_CMAKE_DIR "lib/cmake/paras" CACHE PATH "Install path for CMake config files")
set(PARAS_INSTALL_COMPILER_DIR bin)


# ================== Generate config and version files ==================

configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/paras-config.cmake.in
  ${PROJECT_BINARY_DIR}/paras-config.cmake
  INSTALL_DESTINATION ${PARAS_INSTALL_CMAKE_DIR}
  PATH_VARS
    PARAS_INSTALL_COMPILER_DIR
)

write_basic_package_version_file(
  ${PROJECT_BINARY_DIR}/paras-config-version.cmake
  VERSION ${PARAS_MAJOR_VERSION}
  COMPATIBILITY SameMajorVersion
)

# ================== Install config and version files ==================

install(FILES
  ${PROJECT_BINARY_DIR}/paras-config.cmake
  ${PROJECT_BINARY_DIR}/paras-config-version.cmake
  DESTINATION ${PARAS_INSTALL_CMAKE_DIR}
)

# ================== Export targets ==================

install(EXPORT install_exports
  FILE paras-targets.cmake
  #  NAMESPACE paras::
  DESTINATION ${PARAS_INSTALL_CMAKE_DIR}
)

# ================== Mark advanced ==================

mark_as_advanced(PARAS_INSTALL_CMAKE_DIR)

message(STATUS "ParaSCC CMake configuration installed to ${PARAS_INSTALL_CMAKE_DIR}")

message(STATUS "====================================")
