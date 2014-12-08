# Install script for directory: /home/alox/Tesi/workspace/Dummy/src

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "NativeRel")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "RunTime")
  IF(EXISTS "$ENV{DESTDIR}/opt/gold-4.6/lib/x86_64-linux-gnu/plugins/libgold_app_dummy.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/opt/gold-4.6/lib/x86_64-linux-gnu/plugins/libgold_app_dummy.so")
    FILE(RPATH_CHECK
         FILE "$ENV{DESTDIR}/opt/gold-4.6/lib/x86_64-linux-gnu/plugins/libgold_app_dummy.so"
         RPATH "")
  ENDIF()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/opt/gold-4.6/lib/x86_64-linux-gnu/plugins/libgold_app_dummy.so")
  IF (CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  ENDIF (CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
  IF (CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  ENDIF (CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
FILE(INSTALL DESTINATION "/opt/gold-4.6/lib/x86_64-linux-gnu/plugins" TYPE SHARED_LIBRARY FILES "/home/alox/.config/gold-4.6.0/plugins/libgold_app_dummy.so")
  IF(EXISTS "$ENV{DESTDIR}/opt/gold-4.6/lib/x86_64-linux-gnu/plugins/libgold_app_dummy.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/opt/gold-4.6/lib/x86_64-linux-gnu/plugins/libgold_app_dummy.so")
    FILE(RPATH_REMOVE
         FILE "$ENV{DESTDIR}/opt/gold-4.6/lib/x86_64-linux-gnu/plugins/libgold_app_dummy.so")
    IF(CMAKE_INSTALL_DO_STRIP)
      EXECUTE_PROCESS(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/opt/gold-4.6/lib/x86_64-linux-gnu/plugins/libgold_app_dummy.so")
    ENDIF(CMAKE_INSTALL_DO_STRIP)
  ENDIF()
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "RunTime")

