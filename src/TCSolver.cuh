// #include "src/utils/timer.h"
#include "src/graph/operations.cuh"
#include "src/utils/launcher.h"
using namespace project_AntiRF;

#include "src/gpu_kernels/tc_bs_warp_edge.cuh"
#include "src/gpu_kernels/tc_hi_warp_vertex.cuh"
#include "src/gpu_kernels/tuning_schedules.cuh"
template <typename VID, typename VLABEL>
void TCSolver(project_AntiRF::Graph<VID, VLABEL>& hg, uint64_t& result, int n_dev,
              project_AntiRF::modes cal_m) {
  Stream stream;
  SetDevice(0);
  // size_t nblocks = MAX_GRID_SIZE;
  VID ne = hg.get_enum();
  VID nv = hg.get_vnum();
  // std::cout << "ne is " << ne << std::endl;
  AccType* d_total;
  AccType h_total = 0;
  DMALLOC(d_total, sizeof(AccType));
  TODEV(d_total, &h_total, sizeof(AccType));
  WAIT();
  auto d_g = hg.DeviceObject();
  double time_cost = 0;
  if (cal_m == e_centric) {
    double start = wtime();
    LaunchKernel(
        stream,
        [=] __device__(VID ne, dev::Graph<VID, VLABEL> g, AccType * total) {
          __shared__ typename BlockReduce::TempStorage temp_storage;
          int thread_id = TID_1D;               // global thread index
          int warp_id = thread_id / WARP_SIZE;  // global warp index
          int num_warps =
              WARPS_PER_BLOCK * gridDim.x;  // total number of active warps
          // Test codes: default 2560 warps, 320 blocks
          // if(blockIdx.x == 0 && threadIdx.x == 0) printf("Now process ne %d
          // num_warps %d \n", ne, num_warps); if(blockIdx.x == 0 && threadIdx.x
          // == 0) printf("Num of blocks is %d \n", gridDim.x);
          AccType count = 0;
          for (VID eid = warp_id; eid < ne; eid += num_warps) {
            auto v = g.get_src(eid);
            auto u = g.get_dst(eid);
            // if (v == u) continue; // Note: restriction tested. - not
            // suitable.
            VID v_size = g.getOutDegree(v);
            VID u_size = g.getOutDegree(u);
            // printf("now process eid %d ne %d \n",eid,ne);
            // printf("now process v %d u %d vsize %d usize
            // %d\n",v,u,v_size,u_size); if (v >= u) continue; // // Note:
            // restriction tested. - not suitable.
            count += intersect_num(
                g.getNeighbor(v), v_size, g.getNeighbor(u),
                u_size);  // Note: upper bound tested. - not suitable.
          }
          AccType block_num = BlockReduce(temp_storage).Sum(count);
          if (threadIdx.x == 0)
            atomicAdd(total, block_num);
        },
        ne, d_g, d_total);
    // LaunchKernel(stream, tc_warp_edge<VID, VLABEL>, ne, d_g, d_total);
    stream.Sync();
    double end = wtime();
    time_cost = (end - start);
  } else if (cal_m == v_centric) {
    size_t nthreads = 1024;
    size_t nblocks = 1024;
    size_t bucketnum = BLOCK_BUCKET_NUM;
    size_t bucketsize = TR_BUCKET_SIZE;

    VID* bins;
    auto bins_mem = nblocks * bucketnum * bucketsize * sizeof(VID);
    H_ERR(cudaMalloc((void**) &bins, bins_mem));
    H_ERR(cudaMemset(bins, 0, bins_mem));

    // AccType h_total = 0, *d_total;
    // H_ERR(cudaMalloc((void **)&d_total, sizeof(AccType)));
    // H_ERR(cudaMemcpy(d_total, &h_total, sizeof(AccType),
    // cudaMemcpyHostToDevice)); H_ERR(cudaDeviceSynchronize());

    int nowindex[3];
    int* G_INDEX;

    int block_range = 0;
    //if vertex's degree is large than threshold(USE_CTA), use block to process
    if(true)
    {
      int l = 0, r = nv;
      int val = USE_CTA;
      while (l < r - 1) {
        int mid = (l + r) / 2;
        if (hg.edge_begin(mid + 1) - hg.edge_begin(mid) > val)
          l = mid;
        else
          r = mid;
      }
      if (hg.edge_begin(l + 1) - hg.edge_begin(l) <= val)
        block_range = 0;
      else
        block_range = l + 1;
    }
    nowindex[0] = nblocks * nthreads / WARP_SIZE;
    nowindex[1] = nblocks;
    nowindex[2] =
        block_range + (nblocks * nthreads / WARP_SIZE);
    H_ERR(cudaMalloc((void**) &G_INDEX, sizeof(int) * 3));
    H_ERR(cudaMemcpy(G_INDEX, &nowindex, sizeof(int) * 3,
                     cudaMemcpyHostToDevice));
    double start = wtime();
    tc_hi_warp_vertex<<<nblocks, nthreads>>>(nv, d_g, bins, d_total, G_INDEX,
                                             block_range);
    WAIT();
    double end = wtime();
    time_cost = end - start;
  } else {
    LOG(FATAL) << "Wrong Calculation Mode.";
  }
  LOG(INFO) << "Triangle counting  time: " << time_cost << " seconds";
  TOHOST(d_total, &h_total, sizeof(AccType));
  result = h_total;
  FREE(d_total);
}
