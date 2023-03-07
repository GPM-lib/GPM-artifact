#ifndef UTILS_CUDA_UTILS_H_
#define UTILS_CUDA_UTILS_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>
#include <cub/util_ptx.cuh>
#include <vector>

#include "src/utils/logging.h"
#include "src/utils/utils.h"

namespace project_AntiRF {

void SetDevice(int x) {
  cudaSetDevice(x);
  LOG(INFO) << "Set device to " << x;
}

static void HandleError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    LOG(FATAL) << "CUDA error " << cudaGetErrorString(err) << " at " << file
               << ":" << line;
  }
}

#define H_ERR(err) (HandleError(err, __FILE__, __LINE__))

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define DMALLOC(pdev, bytes) H_ERR(cudaMalloc((void**) &(pdev), (bytes)))
#define TOHOST(pdev, phost, bytes) \
  H_ERR(cudaMemcpy((phost), (pdev), (bytes), D2H))
#define TODEV(pdev, phost, bytes) \
  H_ERR(cudaMemcpy((pdev), (phost), (bytes), H2D))
#define FREE(pdev) H_ERR(cudaFree((pdev)))
#define CLEAN(pdev, bytes) H_ERR(cudaMemset((pdev), 0, (bytes)))
#define WAIT() H_ERR(cudaDeviceSynchronize())

#define DEV_HOST __device__ __host__
#define DEV_HOST_INLINE __device__ __host__ __forceinline__
#define DEV_INLINE __device__ __forceinline__
#define MAX_BLOCK_SIZE 256
#define MAX_GRID_SIZE 768
// #define MAX_BLOCK_SIZE 256
// #define MAX_GRID_SIZE 768
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (MAX_BLOCK_SIZE / WARP_SIZE)
#define TID_1D (threadIdx.x + blockIdx.x * blockDim.x)
#define TOTAL_TID_SIZE (gridDim.x * blockDim.x)

typedef cub::BlockReduce<AccType, MAX_BLOCK_SIZE> BlockReduce;
#define FULL_MASK 0xffffffff

#define ASSERT(x)                                                          \
  if (!(x)) {                                                              \
    LOG(FATAL) << "Assertion failed: " << #x << " at (" << __FILE__ << ":" \
               << __LINE__ << ")";                                         \
  }
}  // namespace project_AntiRF

#if __CUDACC_VER_MAJOR__ >= 9
#define SHFL_DOWN(a, b) __shfl_down_sync(0xFFFFFFFF, a, b)
#define SHFL(a, b) __shfl_sync(0xFFFFFFFF, a, b)
#else
#define SHFL_DOWN(a, b) __shfl_down(a, b)
#define SHFL(a, b) __shfl(a, b)
#endif

#endif  // UTILS_CUDA_UTILS_H_
