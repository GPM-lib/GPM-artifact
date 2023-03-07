#ifndef GRAPH_IO_H
#define GRAPH_IO_H

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

#include "src/graph/graph.h"
#include "src/utils/logging.h"
#include "src/utils/timer.h"

namespace project_AntiRF {

// TODO: support multi-GPU partitioner. and multi-thread.
template <typename VID, typename VLABEL>
class Loader {
 public:
  void Load(std::string const& path, std::string const& format) {
    LOG(INFO) << "load graph from " << path << " in " << format << " format";
    if (format == "mtx")
      _load_mtx(path, true);
    else if (format == "txt")
      _load_txt(path);
    else if (format == "bin")
      _load_bin(path);
    else {
      LOG(FATAL) << "unknown format " << format;
    }
  }

  void LoadVLabel(std::string const& path) {
    LOG(INFO) << "load vertex label from " << path;
    auto isprefix = [](std::string prefix, std::string path) {
      return std::mismatch(prefix.begin(), prefix.end(), path.begin()).first ==
             prefix.end();
    };

    // if from random
    if (isprefix(std::string("random"), path)) {
      size_t range = std::stoi(path.substr(6));
      for (size_t i = 0; i < vsize_; ++i) {
        vlabels_.emplace_back(rand() % range);
      }
      return;
    }

    // if from file
    std::ifstream fin(path);
    if (!fin.is_open()) {
      LOG(FATAL) << "cannot open " << path;
    }
    std::vector<std::pair<VID, VLABEL>> tmp;
    while (fin.good()) {
      VID v;
      VLABEL label;
      fin >> v >> label;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), fin.widen('\n'));
      if (fin.eof())
        break;
      tmp.emplace_back(v, label);
    }
    if (tmp.size() != vsize_) {
      LOG(FATAL) << "vertex size not match";
    }
    vlabels_.resize(vsize_);
    for (auto& p : tmp) {
      vlabels_[p.first - vmin_] = p.second;
    }
  }

  void Build(Graph<VID, VLABEL>& hg) {
    std::vector<VID> vids;
    for (size_t i = 0; i < vsize_; ++i) {
      vids.push_back(i);
    }
    hg.Init(vids, vlabels_, edges_);
  }

 protected:
  void _load_mtx(std::string const& path, bool with_header = false) {
    double start = wtime();
    std::ifstream fin(path);
    if (!fin.is_open()) {
      LOG(FATAL) << "cannot open file " << path;
    }
    // skip comments
    while (1) {
      char c = fin.peek();
      if (c >= '0' && c <= '9')
        break;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), fin.widen('\n'));
    }

    // if with_header == true
    if (with_header) {
      fin >> vsize_ >> vsize_ >> esize_;
    } else {
      vsize_ = esize_ = 0;
    }

    edges_.clear();
    vlabels_.clear();
    vmin_ = std::numeric_limits<VID>::max();
    VID vmax = std::numeric_limits<VID>::min();

    // loop lines
    while (fin.good()) {
      VID v0, v1;
      fin >> v0 >> v1;
      fin.ignore(std::numeric_limits<std::streamsize>::max(), fin.widen('\n'));
      if (fin.eof())
        break;
      if (v0 == v1)
        continue;

      vmin_ = std::min(vmin_, std::min(v0, v1));
      vmax = std::max(vmax, std::max(v0, v1));

      edges_.emplace_back(v0, v1);
      if (fin.eof())
        break;
    }

    vsize_ = vmax - vmin_ + 1;
    esize_ = edges_.size();
    // LOG(INFO) << esize_;
    for (auto& item : edges_) {
      item.first -= vmin_;
      item.second -= vmin_;
    }

    fin.close();
    double end = wtime();
    LOG(INFO) << "load mtx graph in " << (end - start) << " seconds";
  }

  // TODO: support other formats
  void _load_txt(std::string const& path) { LOG(FATAL) << "not implemented"; }
  void _load_bin(std::string const& path) { LOG(FATAL) << "not implemented"; }

 private:
  size_t vsize_;
  size_t esize_;
  VID vmin_;
  std::vector<std::pair<VID, VID>> edges_;
  std::vector<VLABEL> vlabels_;
  bool hasVlabel_ = false;
  // TODO: maybe support elabels_
};

}  // namespace project_AntiRF

#endif  // IO_H
