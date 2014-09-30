cmake_minimum_required(VERSION 2.6)

project(dummy)

# Makefile verbosi
set(CMAKE_VERBOSE_MAKEFILE ON)

# Build a shared library
set(BUILD_SHARED_LIBS ON)

# This block finds the right GOLD installation in any case
# if GOLD is installed, the finder in  ${CMAKE_ROOT} is used
# if GOLD is under development its position can be specified 
# in the GOLD_DIR variable 
if(NOT DEFINED GOLD_DIR AND EXISTS ${CMAKE_ROOT}/Modules/FindGOLD.cmake)
    find_package(GOLD REQUIRED)
elseif(GOLD_DIR)
    set(CMAKE_MODULE_PATH  ${CMAKE_MODULE_PATH} ${GOLD_DIR})
    find_package(GOLD REQUIRED)
else()
    set(GOLD_DIR "" CACHE PATH "GOLD development tree path")
    message("GOLD_DIR not set")
endif()

if(GOLD_FOUND)
    # message("GOLD FOUND ${GOLD_INCLUDE_DIR}")
    # list(REMOVE_ITEM CMAKE_MODULE_PATH ${GOLD_DIR})
    # message("CMAKE_MODULE_PATH=${CMAKE_MODULE_PATH}")

    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${GOLD_MODULE_PATH})

    set( GOLD_BUILD_MODE "Development" CACHE STRING  "Choose the installation type, options are: Development, Package" )
    
    if(BUILD_MODE STREQUAL   "Development")
    else(BUILD_MODE STREQUAL "Package")
    endif()

    configure_file(${GOLD_INCLUDE_DIR}/Framework/Application.h.cmake ${CMAKE_SOURCE_DIR}/src/Application.h)

    include(GOLD_AutoCompilerFlags)
    include(GOLD_FindPackage)

    message("GOLD_RUNTIME_DIR=${GOLD_RUNTIME_DIR}")
    message("GOLD_PLUGINS_DIR=${GOLD_PLUGINS_DIR}")
    message("GOLD_DEV_PLUGINS_DIR=${GOLD_DEV_PLUGINS_DIR}")

    GOLD_FindPackage(Boost REQUIRED FINDER _Boost)

    add_subdirectory(src)
endif()



# Packaging stuff
set(PACKAGE_NAME "dummy")
set(PACKAGE_VERSION_MAJOR "1")
set(PACKAGE_VERSION_MINOR "0")
set(PACKAGE_VERSION_PATCH "0")
set(PACKAGE_VERSION "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}.${PACKAGE_VERSION_PATCH}")
set(PACKAGE_RELEASE "1" )
set(PACKAGE_VENDOR "VisLab" )
set(PACKAGE_TARGET_OS "Linux")
set(COMPONENT "RunTime")
set(PACKAGE_BUILD_DIR ${CMAKE_BINARY_DIR})
# Override default CMAKE_SYSTEM_PROCESSOR
if(UNIX)
  exec_program (uname ARGS -m OUTPUT_VARIABLE PACKAGE_SYSTEM_PROCESSOR )
endif(UNIX)
configure_file( ${CMAKE_SOURCE_DIR}/CPackConfig.cmake ${CMAKE_BINARY_DIR}/CPackConfig.cmake )
include (${CMAKE_BINARY_DIR}/CPackConfig.cmake)
# setup CPack
include( CPack )
