#include <iostream>

#include "src/graph/graph.h"
#include "src/Engine.h"
#include "src/graph/io.h"

using namespace project_AntiRF;

int main(int argc, char *argv[])
{
  if (argc < 3)
  {
    std::cerr << "Usage: " << argv[0] << " <graph_path> <pattern_name>"
              << std::endl;
    return 1;
  }
  std::string path = std::string(argv[1]);
  project_AntiRF::Pattern pattern(argv[2]);
  LOG(INFO) << pattern.get_name()  <<"matching using undirected graphs."
            << "\n";
  Loader<int, int> loader;
  Graph<int, int> hg; // data graph
  algorithms algo = CF;
  modes cal_mode = e_centric; // TODO: support more cal_mode.
  uint64_t result = 0;
  int n_devices = 1; // TODO: support more devices.

  // load data graph
  loader.Load(path, "mtx");
  loader.Build(hg);

  Engine<int, int> engine(hg, n_devices, algo, cal_mode, pattern);
  result = engine.RunPatternEnum();
  
  if(pattern.get_name() != "Motif4")
    LOG(INFO) << "Result: " << result << "\n";
  LOG(INFO) << "Pattern: " << pattern.get_name() << " matching done.";
  return 0;
}