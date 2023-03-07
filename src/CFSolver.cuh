#include "src/graph/operations.cuh"
#include "src/utils/launcher.h"
// #include "omp.h"
using namespace project_AntiRF;

// #include "src/gpu_kernels/clique_4_edge_warp.cuh"
// #include "src/gpu_kernels/clique_5_edge_warp.cuh"
// #include "src/gpu_kernels/clique_6_edge_warp.cuh"
// #include "src/gpu_kernels/clique_7_edge_warp.cuh"
// #include "src/gpu_kernels/clique_8_edge_warp.cuh"

#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"

template <typename VID, typename VLABEL>
__global__ void clique_4_edge_warp(VID ne, dev::Graph<VID, VLABEL> g,
                                   VID* vlists, size_t max_deg,
                                   AccType* total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = TID_1D;
  int warp_id = thread_id / WARP_SIZE;  // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);            // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;      // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;  // total number of active warps
  VID* vlist = &vlists[int64_t(warp_id) * int64_t(max_deg)];
  AccType counter = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK];
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    VID v0_size = g.getOutDegree(v0);
    VID v1_size = g.getOutDegree(v1);
    auto count = intersect(g.getNeighbor(v0), v0_size, g.getNeighbor(v1),
                           v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane] = count;
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane]; i++) {
      VID u = vlist[i];
      VID u_size = g.getOutDegree(u);
      VID v_size = list_size[warp_lane];
      counter += intersect_num(vlist, v_size, g.getNeighbor(u), u_size);
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0)
    atomicAdd(total, block_num);
}

template <typename VID, typename VLABEL>
__global__ void clique_5_edge_warp(VID ne, dev::Graph<VID, VLABEL> g,
                                   VID* vlists, VID max_deg, AccType* total) {
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int thread_id = TID_1D;
  int warp_id = thread_id / WARP_SIZE;  // global warp index
  int thread_lane =
      threadIdx.x & (WARP_SIZE - 1);            // thread index within the warp
  int warp_lane = threadIdx.x / WARP_SIZE;      // warp index within the CTA
  int num_warps = WARPS_PER_BLOCK * gridDim.x;  // total number of active warps
  VID* vlist = &vlists[int64_t(warp_id) * int64_t(max_deg) * 2];
  AccType counter = 0;
  __shared__ VID list_size[WARPS_PER_BLOCK][2];
  for (VID eid = warp_id; eid < ne; eid += num_warps) {
    auto v0 = g.get_src(eid);
    auto v1 = g.get_dst(eid);
    auto v0_size = g.getOutDegree(v0);  // ori: get_degree return VertexID type
    auto v1_size = g.getOutDegree(v1);  // ori: get_degree return VertexID type
    auto count1 = intersect(g.getNeighbor(v0), v0_size, g.getNeighbor(v1),
                            v1_size, vlist);
    if (thread_lane == 0)
      list_size[warp_lane][0] = count1;
    __syncwarp();
    for (VID i = 0; i < list_size[warp_lane][0]; i++) {
      auto v2 = vlist[i];
      auto v2_size =
          g.getOutDegree(v2);  // ori: get_degree return VertexID type
      auto w1_size = list_size[warp_lane][0];
      auto count2 = intersect(vlist, w1_size, g.getNeighbor(v2), v2_size,
                              vlist + max_deg);
      if (thread_lane == 0)
        list_size[warp_lane][1] = count2;
      __syncwarp();
      for (VID j = 0; j < list_size[warp_lane][1]; j++) {
        auto v3 = vlist[max_deg + j];
        auto v3_size =
            g.getOutDegree(v3);  // ori: get_degree return VertexID type
        auto w2_size = list_size[warp_lane][1];
        counter +=
            intersect_num(vlist + max_deg, w2_size, g.getNeighbor(v3), v3_size);
      }
    }
  }
  AccType block_num = BlockReduce(temp_storage).Sum(counter);
  if (threadIdx.x == 0)
    atomicAdd(total, block_num);
}

template <typename VID, typename VLABEL>
void CFSolver(project_AntiRF::Graph<VID, VLABEL>& hg, int k, uint64_t& result,
              int n_dev, project_AntiRF::modes cal_m) {
  ASSERT(k > 3);  // guarantee clique > 4
  // Stream stream;
  SetDevice(0);
  VID ne = hg.get_enum();
  VID max_degree = hg.getMaxDegree();

  size_t per_block_vlist_size =
      WARPS_PER_BLOCK * size_t(k - 3) * size_t(max_degree) * sizeof(VID);

  AccType* d_total;
  AccType h_total = 0;
  DMALLOC(d_total, sizeof(AccType));
  TODEV(d_total, &h_total, sizeof(AccType));
  WAIT();
  auto d_g = hg.DeviceObject();
  int grid_size, block_size;  // uninitialized
  // residency_strategy
  if (1) {
    if (k == 4) {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               clique_4_edge_warp<VID, VLABEL>,
                                               0, (int) MAX_BLOCK_SIZE));
    } else if (k == 5) {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               clique_5_edge_warp<VID, VLABEL>,
                                               0, (int) MAX_BLOCK_SIZE));
    }
  } else {
    // GraphMiner origin.
  }
  size_t flist_size = grid_size * per_block_vlist_size;
  LOG(INFO) << "flist_size is " << flist_size / (1024 * 1024)
            << " MB, grid_size is " << grid_size << ", per_block_vlist_size is "
            << per_block_vlist_size;
  VID* d_frontier_list;
  DMALLOC(d_frontier_list, flist_size);

  if (cal_m == e_centric) {
    if (k == 4) {
      clique_4_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
          ne, d_g, d_frontier_list, max_degree, d_total);
    } else if (k == 5) {
      clique_5_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
          ne, d_g, d_frontier_list, max_degree, d_total);
    } else {
      LOG(FATAL) << "Not supported right now";
    }
    WAIT();  // need wait?
  } else if (cal_m == v_centric) {
    // vertex_centric
  } else {
    LOG(FATAL) << "Wrong Calculation Mode.";
  }
  TOHOST(d_total, &h_total, sizeof(AccType));
  result = h_total;
  FREE(d_total);
}