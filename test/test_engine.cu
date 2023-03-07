#include <iostream>

#include "src/graph/graph.h"
#include "src/Engine.h"
#include "src/graph/io.h"
// #include "src/utils/cuda_utils.h"

using namespace project_AntiRF;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <graph_path> <graph_label>"
              << std::endl;
    return 1;
  }
  std::string path = std::string(argv[1]);
  std::string label_path = std::string(argv[2]); // label --

  Loader<int,int> loader;
  Graph<int,int> hg; // data graph
  algorithms algo = TC; 
  modes cal_mode = e_centric; // TODO: support more cal_mode.
  uint64_t result = 0;
  int n_devices = 1; // TODO: support more devices.

  // load data graph
  loader.Load(path, "mtx");
  loader.LoadVLabel(label_path); // label_path --
  loader.Build(hg);
  // // load query graph
  // loader.Load(query_path, "mtx");
  // loader.LoadVLabel(query_label_path);
  // loader.Build(qg);
  
  Engine<int, int> engine(hg, n_devices, algo, cal_mode);
  result = engine.RunTC();

  LOG(INFO) << "Result: " << result << "\n";
  LOG(INFO) << "Engine test done.";

  return 0;
}