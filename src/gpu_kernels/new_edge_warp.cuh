#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"
#include "tuning_schedules.cuh"
#include "src/graph/search.cuh"
#include "src/graph/set_intersection.cuh"
// #define FREQ_PROFILE
#ifdef FREQ_PROFILE
#define PROFILE(result, size_a, size_b) \
  if (thread_lane == 0)                 \
    (result) += (size_a) * (size_b);
#else
#define PROFILE(result, size_a, size_b)
#endif

#define NPART 1
#define SPACE_PER_V 250
// #define BLOCK_OPT
template <typename VID, typename VLABEL>
__global__ void P1_frequency_count(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType *counters, int *freq_list, unsigned long long *INDEX)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;               // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ VID list_size[WARPS_PER_BLOCK];
  VID count = 0;
  VID calculate_count = 0;
  long long counts[6];
  for (int i = 0; i < 6; i++)
    counts[i] = 0;
  __syncthreads();
  for (VID eid = warp_id; eid < ne;)
  {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    if (v1 == v0)
      continue;
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    VID cnt = 0;

    for (auto i = thread_lane; i < v0_size; i += WARP_SIZE)
    {
      VID key = g.getNeighbor(v0)[i]; // each thread picks a vertex as the key
      int is_smaller = key < v1 ? 1 : 0;
      if (is_smaller && !binary_search(g.getNeighbor(v1), key, v1_size))
        atomicAdd(&freq_list[g.edge_begin(v0) + i], 1);
    }
    PROFILE(calculate_count, v0_size, 1);
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
#ifdef FREQ_PROFILE
  atomicAdd(&counters[3], calculate_count);
#endif
}

template <typename VID, typename VLABEL>
__global__ void P1_pair_freq_count(VID nv, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType *counters, int *freq_list, unsigned long long *INDEX)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;               // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg)];
  __shared__ VID list_size[WARPS_PER_BLOCK];
  long long star3_count = 0;
  VID calculate_count = 0;
  __syncthreads();
  for (VID v0 = warp_id; v0 < nv;)
  {
    __syncwarp();
    VID v0_size = g.getOutDegree(v0);

    for (VID v2_idx = 0; v2_idx < v0_size; v2_idx++)
    {
      VID v2 = g.getNeighbor(v0)[v2_idx];
      VID v2_size = g.getOutDegree(v2);
      VID tmp_cnt = difference_num(g.getNeighbor(v0), v0_size, g.getNeighbor(v2), v2_size, v2);
      VID warp_cnt = warp_reduce<AccType>(tmp_cnt);
      __syncwarp();
      if (thread_lane == 0)
        star3_count += (warp_cnt * freq_list[g.edge_begin(v0) + v2_idx]);
      __syncwarp();
      PROFILE(calculate_count, v0_size, 1);
    }
    NEXT_WORK_CATCH(v0, INDEX, num_warps);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(star3_count);
  if (threadIdx.x == 0)
    atomicAdd(&counters[1], block_num);
#ifdef FREQ_PROFILE
  atomicAdd(&counters[3], calculate_count);
#endif
}

template <typename VID, typename VLABEL>
__global__ void P1_count_correction(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType *counters, int *freq_list, unsigned long long *INDEX)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;               // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
  __shared__ VID list_size[WARPS_PER_BLOCK][2];
  VID calculate_count = 0;
  VID count = 0;
  __syncthreads();
  for (VID eid = warp_id; eid < ne;)
  {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    if (v1 == v0)
      continue;
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto dif_cnt = difference_set(g.getNeighbor(v0), v0_size, g.getNeighbor(v1), v1_size, v1, vlist);
    auto int_cnt = intersect(g.getNeighbor(v0), v0_size, g.getNeighbor(v1), v1_size, v1, &vlist[max_deg]); // y0y1
    if (thread_lane == 0)
    {
      list_size[warp_lane][0] = dif_cnt;
      list_size[warp_lane][1] = int_cnt;
    }
    __syncwarp();
    PROFILE(calculate_count, v0_size, 2);
    for (VID i = 0; i < list_size[warp_lane][1]; i++)
    {
      VID v2 = vlist[max_deg + i];
      VID v2_size = g.getOutDegree(v2);
      for (auto j = thread_lane; j < list_size[warp_lane][0]; j += WARP_SIZE)
      {
        auto key = vlist[j];
        VID key_size = g.getOutDegree(key);
        if (key > v2 && !binary_search(g.getNeighbor(key), v2, key_size))
          count += 1;
      }
    }
    __syncwarp();
    PROFILE(calculate_count, list_size[warp_lane][1], list_size[warp_lane][0]);
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(count);
  if (threadIdx.x == 0)
    atomicAdd(&counters[2], block_num);
#ifdef FREQ_PROFILE
  atomicAdd(&counters[3], calculate_count);
#endif
}

template <typename VID, typename VLABEL>
__global__ void P2_edge_warp(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType *counters, unsigned long long *INDEX)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;               // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  __shared__ VID list_size[WARPS_PER_BLOCK][4];
  VID P2_count = 0;
  long long correct_count = 0;
  VID calculate_count = 0;
  __syncthreads();
  for (VID eid = warp_id; eid < ne;)
  {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);
    auto cnt = difference_set(v0_ptr, v0_size, v1_ptr, v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size, v0_ptr, v0_size, &vlist[max_deg]);
    if (thread_lane == 0)
      list_size[warp_lane][1] = cnt;
    __syncwarp();
    PROFILE(calculate_count, v0_size, 1);
    PROFILE(calculate_count, v1_size, 1);
    for (VID i = thread_lane; i < list_size[warp_lane][0]; i += 32)
    {
      vlist[max_deg * 4 + i] = 0;
    }
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][1]; i++)
    {
      VID v4 = vlist[max_deg + i];
      VID v4_size = g.getOutDegree(v4);
      for (auto j = thread_lane; j < list_size[warp_lane][0]; j += WARP_SIZE)
      {
        VID key = vlist[j];
        if (!binary_search(g.getNeighbor(v4), key, v4_size))
        {
          atomicAdd(&vlist[max_deg * 4 + j], 1);
        }
      }
    }
    __syncwarp();
    PROFILE(calculate_count, list_size[warp_lane][1], list_size[warp_lane][0]);
    for (VID i = 0; i < list_size[warp_lane][0]; i++)
    {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      VID tmp_cnt = difference_num(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, v2);
      VID warp_cnt = warp_reduce<AccType>(tmp_cnt);
      __syncwarp();
      if (thread_lane == 0)
        P2_count += (warp_cnt * vlist[max_deg * 4 + i]);
      __syncwarp();
    }
    __syncwarp();
    PROFILE(calculate_count, list_size[warp_lane][0], list_size[warp_lane][0]);
    for (VID i = 0; i < list_size[warp_lane][1]; i++)
    {
      VID v4 = vlist[max_deg + i];
      VID v4_size = g.getOutDegree(v4);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.getNeighbor(v4), v4_size, &vlist[max_deg * 2]);
      if (thread_lane == 0)
        list_size[warp_lane][2] = cnt;
      __syncwarp();
      cnt = intersect(vlist, list_size[warp_lane][0], g.getNeighbor(v4), v4_size, &vlist[max_deg * 3]);
      if (thread_lane == 0)
        list_size[warp_lane][3] = cnt;
      __syncwarp();
      PROFILE(calculate_count, list_size[warp_lane][0], 2);
      for (VID ii = 0; ii < list_size[warp_lane][3]; ii++)
      {
        VID v2 = vlist[max_deg * 3 + ii];
        VID v2_size = g.getOutDegree(v2);
        for (auto j = thread_lane; j < list_size[warp_lane][2]; j += WARP_SIZE)
        {
          auto key = vlist[max_deg * 2 + j];
          VID key_size = g.getOutDegree(key);
          if (key > v2 && !binary_search(g.getNeighbor(key), v2, key_size))
            correct_count += 1;
        }
      }
      PROFILE(calculate_count, list_size[warp_lane][3], list_size[warp_lane][2]);
    }
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(P2_count);
  AccType block_num1 = BlockReduce(temp_storage).Sum(correct_count);
  if (threadIdx.x == 0)
  {
    atomicAdd(&counters[1], block_num);
    atomicAdd(&counters[2], block_num1);
  }
#ifdef FREQ_PROFILE
  atomicAdd(&counters[3], calculate_count);
#endif
}

template <typename VID, typename VLABEL>
__global__ void P3_edge_warp(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType *counters, unsigned long long *INDEX)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;               // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  __shared__ VID list_size[WARPS_PER_BLOCK][4];
  long long P3_count = 0;
  long long correct_count = 0;
  VID calculate_count = 0;
  __syncthreads();
  for (VID eid = warp_id; eid < ne;)
  {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);
    auto cnt = difference_set(v0_ptr, v0_size, v1_ptr, v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    PROFILE(calculate_count, v0_size, 1);
    for (VID i = thread_lane; i < list_size[warp_lane][0]; i += 32)
    {
      vlist[max_deg * 4 + i] = 0;
    }
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][0]; i++)
    {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      for (auto j = thread_lane; j < i; j += WARP_SIZE)
      {
        VID key = vlist[j];
        if (!binary_search(g.getNeighbor(v2), key, v2_size))
        {
          atomicAdd(&vlist[max_deg * 4 + j], 1);
        }
      }
    }
    __syncwarp();
    PROFILE(calculate_count, list_size[warp_lane][0], list_size[warp_lane][0]);
    for (VID i = 0; i < list_size[warp_lane][0]; i++)
    {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      VID tmp_cnt = difference_num(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, v2);
      VID warp_cnt = warp_reduce<AccType>(tmp_cnt);
      __syncwarp();
      if (thread_lane == 0)
        P3_count += (warp_cnt * vlist[max_deg * 4 + i]);
      __syncwarp();
    }
    __syncwarp();
    PROFILE(calculate_count, list_size[warp_lane][0], list_size[warp_lane][0]);

    for (VID i = 0; i < list_size[warp_lane][0]; i++)
    {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, v2, &vlist[max_deg]);
      if (thread_lane == 0)
        list_size[warp_lane][1] = cnt;
      __syncwarp();
      cnt = intersect(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, &vlist[max_deg * 2]);
      if (thread_lane == 0)
        list_size[warp_lane][2] = cnt;
      __syncwarp();
      PROFILE(calculate_count, list_size[warp_lane][0], 2);
      for (VID ii = 0; ii < list_size[warp_lane][2]; ii++)
      {
        VID v2 = vlist[max_deg * 2 + ii];
        VID v2_size = g.getOutDegree(v2);
        for (auto j = thread_lane; j < list_size[warp_lane][1]; j += WARP_SIZE)
        {
          auto key = vlist[max_deg * 1 + j];
          VID key_size = g.getOutDegree(key);
          if (key > v2 && !binary_search(g.getNeighbor(key), v2, key_size))
            correct_count += 1;
        }
      }
      PROFILE(calculate_count, list_size[warp_lane][2], list_size[warp_lane][1]);
    }
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(P3_count);
  AccType block_num1 = BlockReduce(temp_storage).Sum(correct_count);
  if (threadIdx.x == 0)
  {
    atomicAdd(&counters[1], block_num);
    atomicAdd(&counters[2], block_num1);
  }
#ifdef FREQ_PROFILE
  atomicAdd(&counters[3], calculate_count);
#endif
}

template <typename VID, typename VLABEL>
__global__ void P7_edge_warp(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType *counters, unsigned long long *INDEX)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;               // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 4];
  __shared__ VID list_size[WARPS_PER_BLOCK][4];
  long long P7_count = 0;
  long long correct_count = 0;
  VID calculate_count = 0;
  __syncthreads();
  for (VID eid = warp_id; eid < ne;)
  {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto cnt = intersect(g.getNeighbor(v0), v0_size, g.getNeighbor(v1), v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    PROFILE(calculate_count, v0_size, 1);
    for (VID i = thread_lane; i < list_size[warp_lane][0]; i += 32)
    {
      vlist[max_deg * 3 + i] = 0;
    }
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][0]; i++)
    {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      for (auto j = thread_lane; j < i; j += WARP_SIZE)
      {
        VID key = vlist[j]; // each thread picks a vertex as the key
        if (!binary_search(g.getNeighbor(v2), key, v2_size))
        {
          atomicAdd(&vlist[max_deg * 3 + j], 1);
        }
      }
    }
    __syncwarp();
    PROFILE(calculate_count, list_size[warp_lane][0], list_size[warp_lane][0]);

    for (VID i = 0; i < list_size[warp_lane][0]; i++)
    {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      VID tmp_cnt = difference_num(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, v2);
      VID warp_cnt = warp_reduce<AccType>(tmp_cnt);
      __syncwarp();
      if (thread_lane == 0)
        P7_count += (warp_cnt * vlist[max_deg * 3 + i]);
      __syncwarp();
    }
    PROFILE(calculate_count, list_size[warp_lane][0], list_size[warp_lane][0]);

    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][0]; i++)
    {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, v2, &vlist[max_deg]);
      if (thread_lane == 0)
        list_size[warp_lane][1] = cnt;
      __syncwarp();
      cnt = intersect(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, &vlist[max_deg * 2]);
      if (thread_lane == 0)
        list_size[warp_lane][2] = cnt;
      __syncwarp();
      PROFILE(calculate_count, list_size[warp_lane][0], 1);
      PROFILE(calculate_count, list_size[warp_lane][0], 1);

      for (VID ii = 0; ii < list_size[warp_lane][2]; ii++)
      {
        VID v2 = vlist[max_deg * 2 + ii];
        VID v2_size = g.getOutDegree(v2);
        for (auto j = thread_lane; j < list_size[warp_lane][1]; j += WARP_SIZE)
        {
          auto key = vlist[max_deg + j];
          VID key_size = g.getOutDegree(key);
          if (key > v2 && !binary_search(g.getNeighbor(key), v2, key_size))
            correct_count += 1;
        }
      }
      PROFILE(calculate_count, list_size[warp_lane][2], list_size[warp_lane][1]);
    }
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(P7_count);
  AccType block_num1 = BlockReduce(temp_storage).Sum(correct_count);
  if (threadIdx.x == 0)
  {
    atomicAdd(&counters[1], block_num);
    atomicAdd(&counters[2], block_num1);
  }
#ifdef FREQ_PROFILE
  atomicAdd(&counters[3], calculate_count);
#endif
}

template <typename VID, typename VLABEL>
__global__ void P8_edge_warp(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType *counters, unsigned long long *INDEX)
{
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;               // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  __shared__ VID list_size[WARPS_PER_BLOCK][5];
  long long P8_count = 0;
  long long correct_count = 0;
  VID calculate_count = 0;
  __syncthreads();
  for (VID eid = warp_id; eid < ne;)
  {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);
    auto cnt = intersect(v0_ptr, v0_size, v1_ptr, v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size, v0_ptr, v0_size, &vlist[max_deg]);
    if (thread_lane == 0)
      list_size[warp_lane][1] = cnt;
    __syncwarp();
    PROFILE(calculate_count, v0_size, 1);
    PROFILE(calculate_count, v1_size, 1);
    for (VID i = thread_lane; i < list_size[warp_lane][0]; i += 32)
    {
      vlist[max_deg * 4 + i] = 0;
    }
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][1]; i++)
    {
      VID v4 = vlist[max_deg + i];
      VID v4_size = g.getOutDegree(v4);
      for (auto j = thread_lane; j < list_size[warp_lane][0]; j += WARP_SIZE)
      {
        VID key = vlist[j];
        if (!binary_search(g.getNeighbor(v4), key, v4_size))
        {
          atomicAdd(&vlist[max_deg * 4 + j], 1);
        }
      }
      PROFILE(calculate_count, list_size[warp_lane][0], 1);
    }
    __syncwarp();

    for (VID i = 0; i < list_size[warp_lane][0]; i++)
    {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      VID tmp_cnt = intersect_num(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, v2);
      VID warp_cnt = warp_reduce<AccType>(tmp_cnt);
      __syncwarp();
      if (thread_lane == 0)
        P8_count += (warp_cnt * vlist[max_deg * 4 + i]);
      __syncwarp();
    }
    __syncwarp();
    PROFILE(calculate_count, list_size[warp_lane][0], list_size[warp_lane][0]);
    for (VID i = 0; i < list_size[warp_lane][1]; i++)
    {
      VID v4 = vlist[max_deg + i];
      VID v4_size = g.getOutDegree(v4);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.getNeighbor(v4), v4_size, &vlist[max_deg * 2]);
      if (thread_lane == 0)
        list_size[warp_lane][2] = cnt;
      __syncwarp();
      cnt = intersect(vlist, list_size[warp_lane][0], g.getNeighbor(v4), v4_size, &vlist[max_deg * 3]);
      if (thread_lane == 0)
        list_size[warp_lane][3] = cnt;
      __syncwarp();
      PROFILE(calculate_count, list_size[warp_lane][0], 2);
      for (VID ii = 0; ii < list_size[warp_lane][3]; ii++)
      {
        VID v2 = vlist[max_deg * 3 + ii];
        VID v2_size = g.getOutDegree(v2);
        for (auto j = thread_lane; j < list_size[warp_lane][2]; j += WARP_SIZE)
        {
          auto key = vlist[max_deg * 2 + j];
          VID key_size = g.getOutDegree(key);
          if (key > v2 && binary_search(g.getNeighbor(key), v2, key_size))
            correct_count += 1;
        }
      }

      PROFILE(calculate_count, list_size[warp_lane][3], list_size[warp_lane][2]);
    }
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  AccType block_num = BlockReduce(temp_storage).Sum(P8_count);
  AccType block_num1 = BlockReduce(temp_storage).Sum(correct_count);
  if (threadIdx.x == 0)
  {
    atomicAdd(&counters[1], block_num);
    atomicAdd(&counters[2], block_num1);
  }
#ifdef FREQ_PROFILE
  atomicAdd(&counters[3], calculate_count);
#endif
}

template <typename VID, typename VLABEL>
__global__ void P5_edge_warp(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType *counters, unsigned long long *INDEX, uint32_t *matrix, int width, int col_len)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;               // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  __shared__ VID v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ VID v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
#ifdef BLOCK_OPT
  __shared__ uint32_t blk_map[WARPS_PER_BLOCK][SPACE_PER_V * NPART];
#endif
  VID v2, v2_size;
  VID calculate_count = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][5];
  VID eid = warp_id;
  long long P5_count = 0;
  for (VID eid = warp_id; eid < ne;)
  {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);

    auto cnt = difference_set(v0_ptr, v0_size, v1_ptr, v1_size, vlist); // y0n1
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size, v0_ptr, v0_size, &vlist[max_deg]); // n0y1
    if (thread_lane == 0)
      list_size[warp_lane][1] = cnt;
    __syncwarp();

    cnt = intersect(v0_ptr, v0_size, v1_ptr, v1_size, &vlist[max_deg * 2]); // y0n1
    if (thread_lane == 0)
      list_size[warp_lane][2] = cnt;
    __syncwarp();

    for (VID i = 0; i < list_size[warp_lane][2]; i++)
    {
      VID v2 = vlist[max_deg * 2 + i];
      VID v2_size = g.getOutDegree(v2);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, &vlist[max_deg * 3]); // y0n1
      if (thread_lane == 0)
        list_size[warp_lane][3] = cnt;
      __syncwarp();
      cnt = difference_set(&vlist[max_deg], list_size[warp_lane][1], g.getNeighbor(v2), v2_size, &vlist[max_deg * 4]); // y0n1
      if (thread_lane == 0)
        list_size[warp_lane][4] = cnt;
      __syncwarp();
      if (list_size[warp_lane][3] < list_size[warp_lane][4])
      {
        for (VID j = 0; j < list_size[warp_lane][3]; j++)
        {
          VID v3 = vlist[max_deg * 3 + j];
          VID v3_size = g.getOutDegree(v3);
          if (list_size[warp_lane][4] < v3_size)
          {
// for (auto i = thread_lane; i < list_size[warp_lane][4]; i += WARP_SIZE)
// {
//   auto key = vlist[max_deg * 4 + i];
//   if (!binary_search(g.getNeighbor(v3), key, v3_size))
//     P5_count += 1;
// }
#ifndef BLOCK_OPT
            for (auto k = thread_lane; k < list_size[warp_lane][4]; k += WARP_SIZE)
            {
              auto key = vlist[max_deg * 4 + k];
              if (!binary_search(g.getNeighbor(v3), key, v3_size))
                P5_count += 1;
            }
#endif
#ifdef BLOCK_OPT
            long long addr0 = (long long)(v3) * (long long)col_len;
            for (int i = thread_lane; i < col_len; i += 32)
            {
              blk_map[warp_lane][i] = matrix[addr0 + i];
            }
            __syncwarp();
            for (int i = thread_lane; i < list_size[warp_lane][4]; i += 32)
            {
              VID key = vlist[max_deg * 4 + i];
              auto u_blk = key / width;
              auto u_blk_id = u_blk / 32;
              auto u_blk_oft = u_blk & 31;
              uint32_t bit_ = blk_map[warp_lane][u_blk_id];
              int noneEdge = !(bit_ & (1 << u_blk_oft));
              P5_count += noneEdge;
              if (!noneEdge)
              {
                if (!binary_search(g.getNeighbor(v3), key, v3_size))
                {
                  P5_count += 1;
                }
              }
            }
#endif
          }
          else
          {
            auto tmp_cnt = intersect_num(g.getNeighbor(v3), v3_size, &vlist[max_deg * 4], list_size[warp_lane][4]);
            __syncwarp();
            auto n = warp_reduce<AccType>(tmp_cnt);
            if (thread_lane == 0)
            {
              P5_count += (list_size[warp_lane][4] - n);
            }
          }
        }
      }
      else
      {
        for (VID j = 0; j < list_size[warp_lane][4]; j++)
        {
          VID v4 = vlist[max_deg * 4 + j];
          VID v4_size = g.getOutDegree(v4);
          if (list_size[warp_lane][3] < v4_size)
          {
            // for (auto i = thread_lane; i < list_size[warp_lane][3]; i += WARP_SIZE)
            // {
            //   auto key = vlist[max_deg * 3 + i];
            //   if (!binary_search(g.getNeighbor(v4), key, v4_size))
            //     P5_count += 1;
            // }
#ifndef BLOCK_OPT
            for (auto k = thread_lane; k < list_size[warp_lane][3]; k += WARP_SIZE)
            {
              auto key = vlist[max_deg * 3 + k];
              if (!binary_search(g.getNeighbor(v4), key, v4_size))
                P5_count += 1;
            }
#endif
#ifdef BLOCK_OPT
            long long addr0 = (long long)(v4) * (long long)col_len;
            for (int i = thread_lane; i < col_len; i += 32)
            {
              blk_map[warp_lane][i] = matrix[addr0 + i];
            }
            __syncwarp();
            for (int i = thread_lane; i < list_size[warp_lane][3]; i += 32)
            {
              VID key = vlist[max_deg * 3 + i];
              auto u_blk = key / width;
              auto u_blk_id = u_blk / 32;
              auto u_blk_oft = u_blk & 31;
              uint32_t bit_ = blk_map[warp_lane][u_blk_id];
              int noneEdge = !(bit_ & (1 << u_blk_oft));
              P5_count += noneEdge;
              if (!noneEdge)
              {
                if (!binary_search(g.getNeighbor(v4), key, v4_size))
                {
                  P5_count += 1;
                }
              }
            }
#endif
          }
          else
          {
            auto tmp_cnt = intersect_num(g.getNeighbor(v4), v4_size, &vlist[max_deg * 3], list_size[warp_lane][3]);
            __syncwarp();
            auto n = warp_reduce<AccType>(tmp_cnt);
            if (thread_lane == 0)
            {
              P5_count += (list_size[warp_lane][3] - n);
            }
          }
        }
      }
    }
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  atomicAdd(&counters[1], P5_count);
}
template <typename VID, typename VLABEL>
__global__ void P6_edge_warp(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType *counters, unsigned long long *INDEX, uint32_t *matrix, int width, int col_len)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;               // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 5];
  long long P6_count = 0;
  __shared__ VID v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ VID v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
#ifdef BLOCK_OPT
  __shared__ uint32_t blk_map[WARPS_PER_BLOCK][SPACE_PER_V * NPART];
#endif
  VID v2, v2_size;
  __shared__ VID list_size[WARPS_PER_BLOCK][5];
  for (VID eid = warp_id; eid < ne;)
  {
    VID v0 = g.get_src(eid);
    VID v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto v0_ptr = g.getNeighbor(v0);
    auto v1_ptr = g.getNeighbor(v1);

    auto cnt = intersect(v0_ptr, v0_size, v1_ptr, v1_size, vlist); // y0n1
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size, v0_ptr, v0_size, &vlist[max_deg]); // n0y1
    if (thread_lane == 0)
      list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][0]; i++)
    {
      VID v2 = vlist[i];
      VID v2_size = g.getOutDegree(v2);
      cnt = difference_set(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, &vlist[max_deg * 2]); // y0n1
      if (thread_lane == 0)
        list_size[warp_lane][2] = cnt;
      __syncwarp();
      cnt = difference_set(&vlist[max_deg], list_size[warp_lane][1], g.getNeighbor(v2), v2_size, &vlist[max_deg * 3]); // y0n1
      if (thread_lane == 0)
        list_size[warp_lane][3] = cnt;

      __syncwarp();
      if (list_size[warp_lane][2] < list_size[warp_lane][3])
      {
        for (VID j = 0; j < list_size[warp_lane][2]; j++)
        {
          VID v3 = vlist[max_deg * 2 + j];
          VID v3_size = g.getOutDegree(v3);
          if (list_size[warp_lane][3] < v3_size)
          {
#ifndef BLOCK_OPT
            for (auto k = thread_lane; k < list_size[warp_lane][3]; k += WARP_SIZE)
            {
              auto key = vlist[max_deg * 3 + k];
              if (!binary_search(g.getNeighbor(v3), key, v3_size))
                P6_count += 1;
            }
#endif
#ifdef BLOCK_OPT
            long long addr0 = (long long)(v3) * (long long)col_len;
            for (int i = thread_lane; i < col_len; i += 32)
            {
              blk_map[warp_lane][i] = matrix[addr0 + i];
            }
            __syncwarp();
            for (int i = thread_lane; i < list_size[warp_lane][3]; i += 32)
            {
              VID key = vlist[max_deg * 3 + i];
              auto u_blk = key / width;
              auto u_blk_id = u_blk / 32;
              auto u_blk_oft = u_blk & 31;
              uint32_t bit_ = blk_map[warp_lane][u_blk_id];
              int noneEdge = !(bit_ & (1 << u_blk_oft));
              P6_count += noneEdge;
              if (!noneEdge)
              {
                if (!binary_search(g.getNeighbor(v3), key, v3_size))
                {
                  P6_count += 1;
                }
              }
            }
#endif
          }
          else
          {
            auto tmp_cnt = intersect_num(g.getNeighbor(v3), v3_size, &vlist[max_deg * 3], list_size[warp_lane][3]);
            __syncwarp();
            auto n = warp_reduce<AccType>(tmp_cnt);
            if (thread_lane == 0)
            {
              P6_count += (list_size[warp_lane][3] - n);
            }
          }
        }
      }
      else
      {
        for (VID j = 0; j < list_size[warp_lane][3]; j++)
        {
          VID v3 = vlist[max_deg * 3 + j];
          VID v3_size = g.getOutDegree(v3);
          if (list_size[warp_lane][2] < v3_size)
          {

#ifndef BLOCK_OPT
            for (auto k = thread_lane; k < list_size[warp_lane][2]; k += WARP_SIZE)
            {
              auto key = vlist[max_deg * 2 + k];
              if (!binary_search(g.getNeighbor(v3), key, v3_size))
                P6_count += 1;
            }
#endif
#ifdef BLOCK_OPT
            long long addr0 = (long long)(v3) * (long long)col_len;
            for (int i = thread_lane; i < col_len; i += 32)
            {
              blk_map[warp_lane][i] = matrix[addr0 + i];
            }
            __syncwarp();
            for (int i = thread_lane; i < list_size[warp_lane][2]; i += 32)
            {
              VID key = vlist[max_deg * 2 + i];
              auto u_blk = key / width;
              auto u_blk_id = u_blk / 32;
              auto u_blk_oft = u_blk & 31;
              uint32_t bit_ = blk_map[warp_lane][u_blk_id];
              int noneEdge = !(bit_ & (1 << u_blk_oft));
              P6_count += noneEdge;
              if (!noneEdge)
              {
                if (!binary_search(g.getNeighbor(v3), key, v3_size))
                {
                  P6_count += 1;
                }
              }
            }
#endif
          }
          else
          {
            auto tmp_cnt = intersect_num(g.getNeighbor(v3), v3_size, &vlist[max_deg * 2], list_size[warp_lane][2]);
            __syncwarp();
            auto n = warp_reduce<AccType>(tmp_cnt);
            if (thread_lane == 0)
            {
              P6_count += (list_size[warp_lane][2] - n);
            }
          }
        }
      }
    }
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  atomicAdd(&counters[1], P6_count);
}

template <typename VID, typename VLABEL>
__global__ void P4_edge_warp(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType *counters, unsigned long long *INDEX, uint32_t *matrix, int width, int col_len)

{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;               // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
  __shared__ VID v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ VID v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];
  VID v2, v2_size;
  long long P4_count = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][3];
#ifdef BLOCK_OPT
  __shared__ uint32_t blk_map[WARPS_PER_BLOCK][SPACE_PER_V * NPART];
#endif
  VID eid = warp_id;
  while (eid < ne)
  {
    if (thread_lane == 0)
    {
      v0[warp_lane] = g.get_src(eid);
      v1[warp_lane] = g.get_dst(eid);
    }
    __syncwarp();
    if (v1[warp_lane] >= v0[warp_lane])
    {
      NEXT_WORK_CATCH(eid, INDEX, num_warps);
      continue;
    }
    if (thread_lane == 0)
    {
      v0_size[warp_lane] = g.getOutDegree(v0[warp_lane]);
      v1_size[warp_lane] = g.getOutDegree(v1[warp_lane]);
    }
    __syncwarp();
    auto v0_ptr = g.getNeighbor(v0[warp_lane]);
    auto v1_ptr = g.getNeighbor(v1[warp_lane]);
    VID cnt = 0;
    cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], vlist); // y0n1
    if (thread_lane == 0)
      list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size[warp_lane], v0_ptr, v0_size[warp_lane], &vlist[max_deg]); // n0y1
    if (thread_lane == 0)
      list_size[warp_lane][1] = cnt;
    __syncwarp();

    if (list_size[warp_lane][0] < list_size[warp_lane][1])
      for (VID j = 0; j < list_size[warp_lane][0]; j++)
      {
        v2 = vlist[j];
        v2_size = g.getOutDegree(v2);
        if (list_size[warp_lane][1] < v2_size)
        {
#ifndef BLOCK_OPT
          for (auto i = thread_lane; i < list_size[warp_lane][1]; i += WARP_SIZE)
          {
            auto key = vlist[max_deg + i];
            if (!binary_search(g.getNeighbor(v2), key, v2_size))
              P4_count += 1;
          }
#endif
#ifdef BLOCK_OPT
          long long addr0 = (long long)(v2) * (long long)col_len;
          for (int i = thread_lane; i < col_len; i += 32)
          {
            blk_map[warp_lane][i] = matrix[addr0 + i];
          }
          __syncwarp();
          for (int i = thread_lane; i < list_size[warp_lane][1]; i += 32)
          {
            VID v3 = vlist[i + max_deg];
            auto u_blk = v3 / width;
            auto u_blk_id = u_blk / 32;
            auto u_blk_oft = u_blk & 31;
            uint32_t bit_ = blk_map[warp_lane][u_blk_id];
            int noneEdge = !(bit_ & (1 << u_blk_oft));
            P4_count += noneEdge;
            if (!noneEdge)
            {
              if (!binary_search(g.getNeighbor(v2), v3, v2_size))
              {
                P4_count += 1;
              }
            }
          }
#endif
          __syncwarp();
        }
        else
        {
          auto tmp_cnt = intersect_num(g.getNeighbor(v2), v2_size, &vlist[max_deg], list_size[warp_lane][1]);
          __syncwarp();
          auto n = warp_reduce<AccType>(tmp_cnt);
          if (thread_lane == 0)
          {
            P4_count += (list_size[warp_lane][1] - n);
          }
        }
      }
    else
    {
      for (VID j = 0; j < list_size[warp_lane][1]; j++)
      {
        v2 = vlist[max_deg + j];
        v2_size = g.getOutDegree(v2);
        if (list_size[warp_lane][0] < v2_size)
        {
#ifndef BLOCK_OPT
          for (auto i = thread_lane; i < list_size[warp_lane][0]; i += WARP_SIZE)
          {
            auto key = vlist[i];
            if (!binary_search(g.getNeighbor(v2), key, v2_size))
              P4_count += 1;
          }
#endif
#ifdef BLOCK_OPT
          long long addr0 = (long long)(v2) * (long long)col_len;
          for (int i = thread_lane; i < col_len; i += 32)
          {
            blk_map[warp_lane][i] = matrix[addr0 + i];
          }
          __syncwarp();
          for (int i = thread_lane; i < list_size[warp_lane][0]; i += 32)
          {
            VID v3 = vlist[i];
            auto u_blk = v3 / width;
            auto u_blk_id = u_blk / 32;
            auto u_blk_oft = u_blk & 31;
            uint32_t bit_ = blk_map[warp_lane][u_blk_id];
            int noneEdge = !(bit_ & (1 << u_blk_oft));
            P4_count += noneEdge;
            if (!noneEdge)
            {
              if (!binary_search(g.getNeighbor(v2), v3, v2_size))
              {
                P4_count += 1;
              }
            }
          }
#endif
        }
        else
        {

          auto tmp_cnt = intersect_num(g.getNeighbor(v2), v2_size, vlist, list_size[warp_lane][0]);
          __syncwarp();
          auto n = warp_reduce<AccType>(tmp_cnt);
          if (thread_lane == 0)
          {
            P4_count += (list_size[warp_lane][0] - n);
          }
        }
      }
    }
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }
  atomicAdd(&counters[1], P4_count);
}

__global__ void sub(AccType *accumulators)
{
  accumulators[0] = accumulators[1] - accumulators[2];
  accumulators[1] = 0;
  accumulators[2] = 0;
}

template <typename VID, typename VLABEL>
__global__ void Motif4_edge_warp(VID ne, dev::Graph<VID, VLABEL> g, VID *vlists, VID max_deg, AccType *counters, unsigned long long *INDEX, int *int_maps)
{
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
  int warp_id = thread_id / WARP_SIZE;                   // global warp index
  int thread_lane = threadIdx.x & (WARP_SIZE - 1);       // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;               // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;           // total number of active warps
  VID *vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 4];
  VID counts[6];
  __shared__ VID v0[WARPS_PER_BLOCK], v1[WARPS_PER_BLOCK];
  __shared__ VID v0_size[WARPS_PER_BLOCK], v1_size[WARPS_PER_BLOCK];

  VID *int_map = &int_maps[int64_t(warp_id) * int64_t(max_deg)];

  VID v2, v2_size;
  for (int i = 0; i < 6; i++)
    counts[i] = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][4];

  VID eid = warp_id;
  while (eid < ne)
  {
    if (thread_lane == 0)
    {
      v0[warp_lane] = g.get_src(eid);
      v1[warp_lane] = g.get_dst(eid);
    }
    __syncwarp();
     if (v1[warp_lane] >= v0[warp_lane])
    {
      NEXT_WORK_CATCH(eid, INDEX, num_warps);
      continue;
    }

    if (thread_lane == 0)
    {
      v0_size[warp_lane] = g.getOutDegree(v0[warp_lane]);
      v1_size[warp_lane] = g.getOutDegree(v1[warp_lane]);
    }
    __syncwarp();
    auto v0_ptr = g.getNeighbor(v0[warp_lane]);
    auto v1_ptr = g.getNeighbor(v1[warp_lane]);

    // calculate N(v0)-N(v1) -> vlist
    // N(v0)∩N(v1) -> vlist+max_deg
    // N(v1)-N(v1) -> vlist+max_deg*2
    auto int01_cnt = fuse_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], vlist, &vlist[max_deg], &vlist[max_deg * 2], int_map); // y0y1

    int cnt = 0;
    if (thread_lane == 0)
      list_size[warp_lane][0] = int01_cnt;
    __syncwarp();
    auto dif01_set = vlist;
    auto int01_set = &vlist[max_deg];
    auto dif10_set = &vlist[max_deg * 2];
    auto tmp_set = &vlist[max_deg * 3];
    auto bound_set = &vlist[max_deg * 5];

    for (VID j = 0; j < list_size[warp_lane][0]; j++)
    {
      v2 = int01_set[j];
      v2_size = g.getOutDegree(v2);
      // counting diamond(counts[4]) and 4-clique(counts[5]),
      // v2,v3=N(v0)∩N(v1)
      // if v2-v3 not connect, then count diamond
      // if v2-v3 connect, then count 4-clique
      for (auto i = thread_lane; i < list_size[warp_lane][0]; i += WARP_SIZE)
      {
        auto key = int01_set[i];
        int is_smaller = key < v2 ? 1 : 0;
        // Notice: use direct bs can be fast than shared bs here
        int flag = !binary_search(g.getNeighbor(v2), key, v2_size);
        counts[4] += (flag)&is_smaller;
        counts[5] += ((1 - flag) & is_smaller & (v2 < v1[warp_lane]) & key < v1[warp_lane]);
      }

      // counting tailed(counts[2])
      // v2=N(v0)∩N(v1)
      // v3=N(v2)-N(v0)
      // if v1-v3 not connect, then count tailed
      cnt = difference_set(g.getNeighbor(v2), v2_size, v0_ptr, v0_size[warp_lane], tmp_set); // n0y2
      if (thread_lane == 0)
        list_size[warp_lane][1] = cnt;
      __syncwarp();
      for (auto i = thread_lane; i < list_size[warp_lane][1]; i += WARP_SIZE)
      {
        auto key = tmp_set[i];
        // Notice: use direct bs can be fast than shared bs here
        if (!binary_search(v1_ptr, key, v1_size[warp_lane]))
          counts[2] += 1;
      }
    }
    // counting 4-path(counts[1]) and 4-cycle(counts[3])
    // v2=N(v0)-N(v1)
    // v3=N(v1)-N(v0)
    // if v2-v3 not connect, then count 4-path
    // if v2-v3 connect, then count 4-cycle
    if (thread_lane == 0)
    {
      list_size[warp_lane][0] = v0_size[warp_lane] - int01_cnt;
      list_size[warp_lane][1] = v1_size[warp_lane] - int01_cnt;
    }
    __syncwarp();
    cnt = difference_set(v1_ptr, v1_size[warp_lane], v0_ptr, v0_size[warp_lane], v0[warp_lane], vlist); // n0f0y1
    if (thread_lane == 0) list_size[warp_lane][0] = cnt;
    __syncwarp();
    cnt = difference_set(v0_ptr, v0_size[warp_lane], v1_ptr, v1_size[warp_lane], v1[warp_lane], &vlist[max_deg]); // y0f0n1f1
    if (thread_lane == 0) list_size[warp_lane][1] = cnt;
    __syncwarp();
    for (VID j = 0; j < list_size[warp_lane][1]; j++) {
      v2 = vlist[max_deg+j];
      v2_size = g.getOutDegree(v2);
      counts[3] += intersect_num(vlist, list_size[warp_lane][0], g.getNeighbor(v2), v2_size, v0[warp_lane]); // 4-cycle
    }
    NEXT_WORK_CATCH(eid, INDEX, num_warps);
  }

  for (int i = 0; i < 6; i++)
    atomicAdd(&counters[i], counts[i]);
}