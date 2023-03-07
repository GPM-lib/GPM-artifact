#pragma once

// #include "QueryingOrder.h"
// #include "VerticesFilter.h"
// #include "EnumerationEngine.h"
#include "ds/bitmap.h"
#include "graph/graph.h"
//#include "utils/utils.h"
// #include "src/utils/cuda_utils.h"

#include "TCSolver.cuh"
#include "PatternSolver.cuh"
#include "G2MinerSolver.cuh"
#include "src/utils/sm_pattern.h"

namespace project_AntiRF {
template <typename VID, typename VLABEL>
class Engine {
 private:
  Graph<VID, VLABEL>& hg_;  // data graph
  Graph<VID, VLABEL> qg_;   // query graph ***const
  uint64_t result;
  int n_devices = 1;
  algorithms algo;
  modes cal_mode;  // vertex, edge
  // NOTE: Mix used for MotifCounting and CliqueCounting
  int motif_k = 0;
  std::vector<uint64_t> motif_result;  // WARNING: uninitialized
  project_AntiRF::Pattern s_pattern;
  // NOTE: chunksize? bitmap? // architecture info?
 public:
  Engine(project_AntiRF::Graph<VID, VLABEL>& hg, int n_dev, algorithms al,
         project_AntiRF::modes cal_m)
      : hg_(hg), n_devices(n_dev), algo(al), cal_mode(cal_m) {}
  Engine(project_AntiRF::Graph<VID, VLABEL>& hg, int n_dev, algorithms al,
         project_AntiRF::modes cal_m, int k)
      : hg_(hg), n_devices(n_dev), algo(al), cal_mode(cal_m) {
    motif_k = k;
  }  // motif & clique
  Engine(project_AntiRF::Graph<VID, VLABEL>& hg, int n_dev, algorithms al,
         project_AntiRF::modes cal_m, project_AntiRF::Pattern p)
      : hg_(hg), n_devices(n_dev), algo(al), cal_mode(cal_m) {
    s_pattern = p;  // pattern
  }                 // subgraph matching

  // Note: remind to reconstruct, algo is expired now.
  uint64_t RunTC() {
    // orientation here
    if (cal_mode == v_centric) {
      hg_.orientation(false);
      hg_.SortCSRGraph(true);
    } else
      hg_.orientation();
    TCSolver(hg_, result, n_devices, cal_mode);
    return result;
  }

  uint64_t RunPatternEnum() {
    PatternSolver(hg_, result, n_devices, cal_mode, s_pattern);
    return result;
  }

  uint64_t RunG2Miner() {
    G2MinerSolver(hg_, result, n_devices, cal_mode, s_pattern);
    return result;
  }

  //~Engine()
};
}  // namespace project_AntiRF
