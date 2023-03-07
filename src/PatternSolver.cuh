#include "src/graph/operations.cuh"
#include "src/utils/launcher.h"
// #include "omp.h"
using namespace project_AntiRF;

// #include "src/gpu_kernels/clique_4_edge_warp.cuh"
// #include "src/gpu_kernels/pattern_enum.cuh"
// #include "src/gpu_kernels/clique_6_edge_warp.cuh"
// #include "src/gpu_kernels/clique_7_edge_warp.cuh"
// #include "src/gpu_kernels/clique_8_edge_warp.cuh"

#include "src/gpu_kernels/new_edge_warp.cuh"
#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"
#include "src/utils/sm_pattern.h"
template <typename VID, typename VLABEL>
void PatternSolver(project_AntiRF::Graph<VID, VLABEL> &hg, uint64_t &result, int n_dev,
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
  uint32_t *matrix;
  int width = 1000;
  int block_per_v = (nv + width - 1) / width;
  int bitset_per_v = (block_per_v + 32 - 1) / 32;
  int global_idx = 0;
  long long col_len = (long long)(bitset_per_v);
  long long row_len = (long long)(nv);
  {
    if (p.get_name() == "P1")
    {

      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P1_count_correction<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P2")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P2_edge_warp<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P3")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P3_edge_warp<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P4")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P4_edge_warp<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P5")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P5_edge_warp<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P6")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P6_edge_warp<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P7")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P7_edge_warp<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "P8")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P8_edge_warp<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else if (p.get_name() == "Motif4")
    {
      H_ERR(cudaOccupancyMaxPotentialBlockSize(&grid_size, &block_size,
                                               P1_count_correction<VID, VLABEL>, 0,
                                               (int)MAX_BLOCK_SIZE));
    }
    else
    {
      LOG(FATAL) << p.get_name() << " not support!";
    }
  }

#ifdef BLOCK_OPT

  long long nv_nv = (long long)((long long)row_len * (long long)col_len);
  uint32_t *h_matrix = (uint32_t *)malloc(sizeof(uint32_t) * nv_nv);

  DMALLOC(matrix, sizeof(uint32_t) * nv_nv);
  CLEAN(matrix, sizeof(uint32_t) * nv_nv);
  {
    for (int v = 0; v < nv; v += 1)
    {
      auto v_ptr = hg.getNeighbor(v);
      auto v_size = hg.getOutDegree(v);
      auto v_blk = v / width;
      auto v_blk_id = v_blk / 32;
      auto v_blk_oft = v_blk & 31;
      int pre = -1;
      int local_idx = 0;
      for (int j = 0; j < v_size; j++)
      {
        auto u = v_ptr[j];
        auto u_blk = u / width;
        auto u_oft = u % width;

        if (u_blk != pre)
        {
          local_idx++;
          pre = u_blk;
        }

        auto u_blk_id = u_blk / 32;
        auto u_blk_oft = u_blk & 31;
        long long addr0 = (long long)(v) * (long long)col_len + (long long)(u_blk_id);
        long long addr1 = (long long)(u) * (long long)col_len + (long long)(v_blk_id);
        h_matrix[addr0] |= (1 << u_blk_oft);
        h_matrix[addr1] |= (1 << v_blk_oft);
      }
    }
  }
  TODEV(matrix, h_matrix, sizeof(uint32_t) * nv_nv);
  printf("create matrix done!!\n");
  std::cout << "matrix  size: " << nv_nv * sizeof(uint32_t) / (1024 * 1024) << " MB\n";

#endif
  unsigned long long *G_INDEX, *G_INDEX1, *G_INDEX2;
  unsigned long long nowindex = (grid_size * block_size / WARP_SIZE);
  DMALLOC(G_INDEX, sizeof(unsigned long long));
  DMALLOC(G_INDEX1, sizeof(unsigned long long));
  DMALLOC(G_INDEX2, sizeof(unsigned long long));
  TODEV(G_INDEX, &nowindex, sizeof(unsigned long long));
  TODEV(G_INDEX1, &nowindex, sizeof(unsigned long long));
  TODEV(G_INDEX2, &nowindex, sizeof(unsigned long long));

  int *int_map;
  auto n_size = grid_size * WARPS_PER_BLOCK * size_t(max_degree);
  DMALLOC(int_map, n_size * sizeof(VID));

  size_t flist_size = grid_size * per_block_vlist_size;
  LOG(INFO) << "flist_size is " << flist_size / (1024 * 1024)
            << " MB, grid_size is " << grid_size << ", per_block_vlist_size is "
            << per_block_vlist_size;
  VID *d_frontier_list;
  int *freq_list;
  DMALLOC(d_frontier_list, flist_size);
  if (p.get_name() == "P1" || p.get_name() == "Motif4")
  {
    DMALLOC(freq_list, ne * sizeof(int));
    CLEAN(freq_list, ne * sizeof(int));
  }
  double start = wtime();
  if (p.get_name() == "P1")
  {
    P1_frequency_count<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, freq_list, G_INDEX);
    P1_pair_freq_count<VID, VLABEL><<<grid_size, block_size>>>(
        nv, d_g, d_frontier_list, max_degree, d_total, freq_list, G_INDEX1);
    P1_count_correction<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, freq_list, G_INDEX2);
  }
  else if (p.get_name() == "P2")
  {
    P2_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, G_INDEX);
  }
  else if (p.get_name() == "P3")
  {
    P3_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, G_INDEX);
  }
  else if (p.get_name() == "P4")
  {
    P4_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, G_INDEX, matrix, width, col_len);
  }
  else if (p.get_name() == "P5")
  {
    P5_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, G_INDEX, matrix, width, col_len);
  }
  else if (p.get_name() == "P6")
  {
    P6_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, G_INDEX, matrix, width, col_len);
  }
  else if (p.get_name() == "P7")
  {
    P7_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, G_INDEX);
  }
  else if (p.get_name() == "P8")
  {
    P8_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, G_INDEX);
  }
  else if (p.get_name() == "Motif4")
  {

    P1_frequency_count<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, freq_list, G_INDEX);
    P1_pair_freq_count<VID, VLABEL><<<grid_size, block_size>>>(
        nv, d_g, d_frontier_list, max_degree, d_total, freq_list, G_INDEX1);
    P1_count_correction<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, freq_list, G_INDEX2);
    sub<<<1, 1>>>(d_total);
    WAIT();
    CLEAN(G_INDEX, sizeof(unsigned long long));
    CLEAN(G_INDEX1, sizeof(unsigned long long));
    WAIT();
    Motif4_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, G_INDEX, int_map);
    P4_edge_warp<VID, VLABEL><<<grid_size, block_size>>>(
        ne, d_g, d_frontier_list, max_degree, d_total, G_INDEX1, matrix, width, col_len);
  }
  else
  {
    LOG(FATAL) << p.get_name() << " not support!";
  }

  WAIT();
  double end = wtime();
  LOG(INFO) << p.get_name() << " matching  time: " << (end - start) << " seconds";

  TOHOST(d_total, &h_total, count_length * sizeof(AccType));
  if (p.get_name() == "P4" || p.get_name() == "P5" || p.get_name() == "P6")
    result = h_total[1];
  else
    result = h_total[1] - h_total[2];

  if (p.get_name() == "Motif4")
  {
    for (int i = 0; i < count_length; i++)
    {
      LOG(INFO) << "Result" << i << ": " << h_total[i] << "\n";
    }
  }
  // result = h_total[1];
  FREE(d_total);
}