#!/bin/sh
# Automatic build of the rockwell packages
# This script must be run from the source directory
###############################################################################

BUILD_TYPE='NativeRel'
PACKAGE_NAME='dummy'
PACKAGE_VERSION_MAJOR=1
PACKAGE_VERSION_MINOR=0
PACKAGE_VERSION_PATCH=0
PACKAGE_RELEASE=1
PACKAGE_TARGET_OS=fc14
# INSTALL_PREFIX=/opt/gold-4.4.0
# GOLD_DIR=/put_your_gold_build_dir_here
# GOLD_DIR=/media/sdb2/Development/gold-trunk/build_package
BUILD_DIR="build_package"

# creates and go into the package build directory 
rm -fr ${BUILD_DIR}
mkdir ${BUILD_DIR}
cd ${BUILD_DIR}

# Setup for parallel builds
numprocs=`egrep -c ^cpu[0-9]+ /proc/stat || :`
if [ "$numprocs" = "0" ]; then
  numprocs=1
fi
echo Building package using $numprocs processes

cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} .. 
cpack -G RPM --config CPackConfig.cmake

# roll back initialization files to original name
cd ..

