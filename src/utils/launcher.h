#ifndef UTILS_LAUNCHER_H
#define UTILS_LAUNCHER_H

#include <cub/grid/grid_barrier.cuh>

#include "src/utils/stream.h"

namespace project_AntiRF {

namespace detail {

template <typename F, typename... Args>
void for_each_argument_address(F f, Args&&... args) {
  [](...) {}((f((void*) &std::forward<Args>(args)), 0)...);
}

template <typename KernelFunction, typename... KernelParameters>
inline void cooperative_launch(const KernelFunction& kernel_function,
                               cudaStream_t stream, dim3 grid_dims,
                               dim3 block_dims,
                               KernelParameters&&... parameters) {
  void* arguments_ptrs[sizeof...(KernelParameters)];
  auto arg_index = 0;
  detail::for_each_argument_address(
      [&](void* x) { arguments_ptrs[arg_index++] = x; }, parameters...);
  H_ERR(cudaLaunchCooperativeKernel<KernelFunction>(
      &kernel_function, grid_dims, block_dims, arguments_ptrs, 0, stream));
}

template <typename KernelFunction, typename... KernelParameters>
inline void cooperative_multi_device_launch(
    int num_device, const KernelFunction& kernel_function, cudaStream_t stream,
    dim3 grid_dims, dim3 block_dims, KernelParameters&&... parameters) {
  void* arguments_ptrs[sizeof...(KernelParameters)];
  auto arg_index = 0;
  detail::for_each_argument_address(
      [&](void* x) { arguments_ptrs[arg_index++] = x; }, parameters...);

  struct cudaLaunchParams params[num_device];
  for (int i = 0; i < num_device; i++) {
    params[i].func = (void*) &kernel_function;
    params[i].gridDim = grid_dims;  // Use occupancy calculator
    params[i].blockDim = block_dims;
    params[i].sharedMem = 0;
    params[i].stream = stream;  // Cannot use the NULL stream
    params[i].args = arguments_ptrs;
  }
  H_ERR(cudaLaunchCooperativeKernelMultiDevice(params, num_device));
}
}  // namespace detail

template <typename F, typename... Args>
__global__ void KernelWrapper(F f, Args... args) {
  f(args...);
}

template <typename F, typename... Args>
__global__ void KernelWrapperWithBarrier(cub::GridBarrier barrier, F f,
                                         Args... args) {
  f(barrier, args...);
}

template <typename F, typename... Args>
__global__ void KernelWrapperFused(cub::GridBarrier barrier, F f,
                                   Args... args) {
  bool running;
  do {
    running = f(barrier, args...);
    barrier.Sync();
  } while (running);
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, F f, Args&&... args) {
  int grid_size, block_size;

  H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                           KernelWrapper<F, Args...>, 0,
                                           (int) MAX_BLOCK_SIZE));

  // KernelWrapper<<<1, 1, 0, stream.cuda_stream()>>>(
  //     f, std::forward<Args>(args)...);
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchCooperativeKernel(const Stream& stream, F f, Args&&... args) {
  int grid_size, block_size;
  H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                           KernelWrapper<F, Args...>, 0,
                                           (int) MAX_BLOCK_SIZE));
  dim3 grid_dims(grid_size), block_dims(block_size);
  detail::cooperative_launch(KernelWrapper<F, Args...>, stream.cuda_stream(),
                             grid_dims, block_dims, f, args...);
}

template <typename F, typename... Args>
void LaunchCooperativeMultiDeviceKernel(int num_device, const Stream& stream,
                                        F f, Args&&... args) {
  int grid_size, block_size;
  H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                           KernelWrapper<F, Args...>, 0,
                                           (int) MAX_BLOCK_SIZE));
  dim3 grid_dims(grid_size), block_dims(block_size);
  detail::cooperative_multi_device_launch(num_device, KernelWrapper<F, Args...>,
                                          stream.cuda_stream(), grid_dims,
                                          block_dims, f, args...);
}
template <typename SIZE_T>
inline int round_up(SIZE_T work_size, int& block_size) {
  return (int) work_size > block_size ? work_size : block_size;
}

template <typename SIZE_T>
inline void KernelSizing(int& block_num, int& block_size, SIZE_T work_size) {
  block_size = MAX_BLOCK_SIZE;
  block_num = std::min(MAX_GRID_SIZE, (int) round_up(work_size, block_size));
}

template <typename F, typename... Args>
void LaunchKernel(const Stream& stream, size_t size, F f, Args&&... args) {
  int grid_size, block_size;

  KernelSizing(grid_size, block_size, size);
  KernelWrapper<<<grid_size, block_size, 0, stream.cuda_stream()>>>(
      f, std::forward<Args>(args)...);
}

template <typename F, typename... Args>
void LaunchKernelWithBarrier(const Stream& stream, F f, Args&&... args) {
  int dev;
  int work_residency;
  cudaDeviceProp props{};
  int block_size = MAX_GRID_SIZE;

  H_ERR(cudaGetDevice(&dev));
  H_ERR(cudaGetDeviceProperties(&props, dev));

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &work_residency, KernelWrapperWithBarrier<F, Args...>, block_size, 0);

  size_t grid_size = props.multiProcessorCount * work_residency;
  dim3 grid_dims(grid_size), block_dims(block_size);
  cub::GridBarrierLifetime barrier;

  barrier.Setup(grid_size);
  KernelWrapperWithBarrier<<<grid_dims, block_dims>>>(
      barrier, f, std::forward<Args>(args)...);
}

}  // namespace project_AntiRF
#endif  // UTILS_LAUNCHER_H
