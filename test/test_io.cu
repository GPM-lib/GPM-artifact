#include <iostream>
#include <string>

#include "src/graph/graph.h"
#include "src/graph/io.h"

using namespace project_AntiRF;

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <graph_path> <label_path>"
              << std::endl;
    return 1;
  }
  std::string path = std::string(argv[1]);
  std::string label_path = std::string(argv[2]);

  Loader<int,int> loader;
  Graph<int,int> hg;

  loader.Load(path, /*format=*/"mtx");
  loader.LoadVLabel(label_path);
  loader.Build(hg);
  hg.Dump(std::cout);
  hg.DumpCOO(std::cout);
  LOG(INFO) << "IO test done.";

  return 0;
}
