# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

# compile CUDA with /usr/local/cuda-11.7/bin/nvcc
CUDA_DEFINES = 

CUDA_INCLUDES = -I/home/linzhiheng/shared/GPM-artifact/thirdparty -I/home/linzhiheng/shared/GPM-artifact -I/usr/local/cuda-11.7/include -isystem=/home/linzhiheng/shared/GPM-artifact/thirdparty/cub -isystem=/home/linzhiheng/shared/GPM-artifact/thirdparty/thrust -isystem=/home/linzhiheng/shared/GPM-artifact/thirdparty/moderngpu/src -isystem=/home/linzhiheng/dep/openmpi/include

CUDA_FLAGS =  --expt-extended-lambda -rdc=true -O3 -DNDEBUG --generate-code=arch=compute_75,code=[compute_75,sm_75] -std=c++14
