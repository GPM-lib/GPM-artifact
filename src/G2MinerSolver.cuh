#include "src/graph/operations.cuh"
#include "src/utils/launcher.h"
// #include "omp.h"
using namespace project_AntiRF;

// #include "src/gpu_kernels/clique_4_edge_warp.cuh"
// #include "src/gpu_kernels/pattern_enum.cuh"
// #include "src/gpu_kernels/clique_6_edge_warp.cuh"
// #include "src/gpu_kernels/clique_7_edge_warp.cuh"
// #include "src/gpu_kernels/clique_8_edge_warp.cuh"

#include "src/gpu_kernels/G2Miner_kernels.cuh"
#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"
#include "src/Engine.h"
#include "src/utils/sm_pattern.h"
template <typename VID, typename VLABEL>
void G2MinerSolver(project_AntiRF::Graph<VID, VLABEL> &hg, uint64_t &result, int n_dev,
                   project_AntiRF::modes cal_m, project_AntiRF::Pattern p)
{
  // Stream stream;
  SetDevice(0);
  VID ne = hg.get_enum();
  VID nv = hg.get_vnum();
  VID max_degree = hg.getMaxDegree();

  int list_num = 6;
  size_t per_block_vlist_size =
      // WARPS_PER_BLOCK * size_t(k - 3) * size_t(max_degree) * sizeof(VID);
      WARPS_PER_BLOCK * list_num * size_t(max_degree) * sizeof(VID);

  AccType *d_total;
  int count_length = 6;
  AccType h_total[count_length] = {0};
  DMALLOC(d_total, count_length * sizeof(AccType));
  TODEV(d_total, &h_total, count_length * sizeof(AccType));
  // CLEAN(d_total, 4*sizeof(AccType));
  WAIT();
  auto d_g = hg.DeviceObject();
  int grid_size, block_size; // uninitialized

  {
    if (p.get_name() == "P1")
    {

      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P1_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P2")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P2_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P3")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P3_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P4")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P4_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P5")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P5_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P6")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P6_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P7")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P7_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P8")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P8_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "Motif4")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               Motif4_G2Miner<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else
    {
      LOG(FATAL) << p.get_name() << " not support!";
    }
  }
  size_t flist_size = grid_size * per_block_vlist_size;
  LOG(INFO) << "flist_size is " << flist_size / (1024 * 1024)
            << " MB, grid_size is " << grid_size << ", per_block_vlist_size is "
            << per_block_vlist_size;
  VID *d_frontier_list;
  DMALLOC(d_frontier_list, flist_size);

  double start = wtime();
  if (p.get_name() == "P1")
  {
    P1_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  }
  else if (p.get_name() == "P2")
  {
    P2_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  }
  else if (p.get_name() == "P3")
  {
    P3_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  }
  else if (p.get_name() == "P4")
  {
    P4_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  }
  else if (p.get_name() == "P5")
  {
    P5_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  }
  else if (p.get_name() == "P6")
  {
    P6_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  }
  else if (p.get_name() == "P7")
  {
    P7_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  }
  else if (p.get_name() == "P8")
  {
    P8_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  }
  else if (p.get_name() == "Motif4")
  {
    Motif4_G2Miner<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total);
  }
  else
  {
    LOG(FATAL) << p.get_name() << " not support!";
  }

  WAIT();
  double end = wtime();
  LOG(INFO) << p.get_name() << " matching  time: " << (end - start) << " seconds";

  TOHOST(d_total, &h_total, count_length * sizeof(AccType));
  result = h_total[0];
  if (p.get_name() == "Motif4")
  {
    for (int i = 0; i < count_length; i++)
    {
      LOG(INFO) << "Result" << i << ": " << h_total[i] << "\n";
    }
  }

  FREE(d_total);
}