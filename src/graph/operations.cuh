#ifndef GRAPH_OPERATIONS_H
#define GRAPH_OPERATIONS_H
#include "src/graph/set_difference.cuh"
#include "src/graph/set_intersection.cuh"
#include "src/utils/cuda_utils.h"

template <typename T>
DEV_INLINE T warp_reduce(T val) {
  T sum = val;
  sum += SHFL_DOWN(sum, 16);
  sum += SHFL_DOWN(sum, 8);
  sum += SHFL_DOWN(sum, 4);
  sum += SHFL_DOWN(sum, 2);
  sum += SHFL_DOWN(sum, 1);
  sum = SHFL(sum, 0);
  return sum;
}

template <typename T>
DEV_INLINE void warp_reduce_iterative(T& val) {
  for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  val = SHFL(val, 0);
}

template <typename T>
DEV_INLINE unsigned count_smaller(T bound, T* a, T size_a) {
  if (size_a == 0)
    return 0;
  unsigned thread_lane =
      threadIdx.x & (WARP_SIZE - 1);             // thread index within the warp
  unsigned warp_lane = threadIdx.x / WARP_SIZE;  // warp index within the CTA
  __shared__ unsigned count[WARPS_PER_BLOCK];
  __shared__ unsigned begin[WARPS_PER_BLOCK];
  __shared__ unsigned end[WARPS_PER_BLOCK];
  if (thread_lane == 0) {
    count[warp_lane] = 0;
    begin[warp_lane] = 0;
    end[warp_lane] = size_a;
  }
  __syncwarp();
  bool found = false;
  int mid = 0;
  int l = begin[warp_lane];
  int r = end[warp_lane];
  while (r - l > 32 * 4) {
    mid = l + (r - l) / 2;
    auto value = a[mid];
    if (value == bound) {
      found = true;
      break;
    }
    if (thread_lane == 0) {
      if (value < bound)
        begin[warp_lane] = mid + 1;
      else
        end[warp_lane] = mid - 1;
    }
    __syncwarp();
    l = begin[warp_lane];
    r = end[warp_lane];
  }
  if (found)
    return mid;
  if (thread_lane == 0)
    count[warp_lane] = begin[warp_lane];
  for (auto i = thread_lane + l; i < r; i += WARP_SIZE) {
    int found = 0;
    if (a[i] < bound)
      found = 1;
    unsigned active = __activemask();
    unsigned mask = __ballot_sync(active, found);
    if (thread_lane == 0)
      count[warp_lane] += __popc(mask);
    __syncwarp(active);
    if (mask != FULL_MASK)
      break;
  }
  return count[warp_lane];
}
#endif