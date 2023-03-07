project(PROJECT_ANTIRF LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.0)

option(BUILD_TESTS "Build tests" ON)

# ------------------------------------------------------------------------------
# setting
# ------------------------------------------------------------------------------
include_directories(thirdparty)
include_directories(thirdparty/cub)
include_directories(thirdparty/thrust)
include_directories(thirdparty/moderngpu/src)

include(CheckCXXCompilerFlag)
include(CheckLibraryExists)
include(ExternalProject)
include(GNUInstallDirs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(DEFAULT_BUILD_TYPE "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE
    STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()

message("[Project_AntiRF] will build in type: ${CMAKE_BUILD_TYPE}")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -Wno-unknown-pragmas -fmax-errors=1 -fopenmp -Werror -Wl,-rpath,$ORIGIN")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fprofile-arcs -ftest-coverage")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -g")

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -rdc=true")
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 61 70)
endif ()
message("CUDA ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

# ------------------------------------------------------------------------------
# find_libraries
# ------------------------------------------------------------------------------
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})

# find glog---------------------------------------------------------------------
include("cmake/FindGlog.cmake")
if (NOT GLOG_FOUND)
  message(FATAL_ERROR "glog not found, please install the glog library")
else ()
  include_directories(BEFORE SYSTEM ${GLOG_INCLUDE_DIRS})
endif ()

# find gflags-------------------------------------------------------------------
include("cmake/FindGFlags.cmake")

if (NOT GFLAGS_FOUND)
  message(STATUS "gflags not found, build without gflags")
else ()
  include_directories(BEFORE SYSTEM ${GFLAGS_INCLUDE_DIRS})
endif ()

# find Threads------------------------------------------------------------------
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)

# find CUDA---------------------------------------------------------------------
find_package(CUDA 9 REQUIRED)
include_directories(${CUDA_TOOLKIT_INCLUDE})
set(CUDA_LIBS ${CUDA_TOOLKIT_TARGET_DIR}/lib64/stubs/libcuda.so
  ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libnvToolsExt.so
  ${CUDA_TOOLKIT_TARGET_DIR}/lib64/libcudart.so)

if(${CMAKE_VERSION} VERSION_LESS_EQUAL "3.13.4")
	#cuda_select_nvcc_arch_flags(ARCH_FLAGS "Auto")
  set(ARCH_FLAGS "75")
  message(STATUS "CUDA_ARCH = ${ARCH_FLAGS}")
  string(REPLACE "-gencode;" "--generate-code=" ARCH_FLAGS "${ARCH_FLAGS}")
  string(APPEND CMAKE_CUDA_FLAGS "${ARCH_FLAGS}")
else()
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  set(CUDA_ARCH_LIST "75")
  SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  message(STATUS "CUDA_ARCH = ${CUDA_ARCH_LIST}")
  set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
endif()
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-g;-lineinfo;-Xcompiler;-ggdb;-std=c++14;--expt-extended-lambda)
else ()
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-std=c++14;--expt-extended-lambda)
endif ()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION OFF)
message(STATUS "Host Compiler: ${CUDA_HOST_COMPILER}")



# find NCCL---------------------------------------------------------------------
# 2.7 at least (we fouces on single gpu now)
#include("cmake/FindNCCL.cmake")
#include_directories(SYSTEM ${NCCL_INCLUDE_DIRS})
#if (NCCL_MAJOR_VERSION VERSION_LESS "2" OR NCCL_MINOR_VERSION VERSION_LESS "7")
#  message(FATAL_ERROR "The version of NCCL is required at least 2.7")
#endif ()

# ------------------------------------------------------------------------------
# generate target
# ------------------------------------------------------------------------------

# no library now
#file(GLOB_RECURSE CORE_SRC_FILES "src/*.cc")
#add_library(AntiRF ${CORE_SRC_FILES})
#
#target_link_libraries(AntiRF ${MPI_CXX_LIBRARIES}
#  ${CMAKE_THREAD_LIBS_INIT}
#  ${GLOG_LIBRARIES})

add_custom_target(tests)
if(BUILD_TESTS)
  file(GLOB TEST_FILES RELATIVE "${PROJECT_SOURCE_DIR}/test" "${PROJECT_SOURCE_DIR}/test/*.cu")
  foreach(f ${TEST_FILES})
    string(REGEX MATCH "^(.*)\\.[^.]*$" dummy ${f})
    set(T_NAME ${CMAKE_MATCH_1})
    message(STATUS "Found unit_test - " ${T_NAME})
    cuda_add_executable(${T_NAME} test/${T_NAME}.cu)
    target_include_directories(${T_NAME} SYSTEM PRIVATE thirdparty/cub thirdparty/thrust thirdparty/moderngpu/src)
    set_target_properties(${T_NAME} PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(${T_NAME} ${MPI_CXX_LIBRARIES}
      ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${GTEST_LIBRARY}
      ${CMAKE_DL_LIBS} ${NCCL_LIBRARIES})
    add_dependencies(tests ${T_NAME})
  endforeach()
endif()

# ------------------------------------------------------------------------------
# format code
# ------------------------------------------------------------------------------
file(GLOB_RECURSE FILES_NEED_FORMAT "src/*.cc"
  "src/*.cu"
  "src/*.h"
  "tests/*.h"
  "tests/*.cc"
  "tests/*.cu")
foreach (file_path ${FILES_NEED_FORMAT})
  if (${file_path} MATCHES ".*thirdparty.*")
    list(REMOVE_ITEM FILES_NEED_FORMAT ${file_path})
  endif ()
endforeach ()

add_custom_target(clformat
  COMMAND clang-format --style=file -i ${FILES_NEED_FORMAT}
  COMMENT "Running clang-format."
  VERBATIM)
