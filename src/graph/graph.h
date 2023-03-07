#ifndef GRAPH_GRAPH_H
#define GRAPH_GRAPH_H

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "src/graph/operations.cuh"
#include "src/graph/scan.h"
#include "src/utils/buffer.h"
#include "src/utils/cuda_utils.h"
#include "src/utils/logging.h"
#include "src/utils/timer.h"
#include "src/utils/utils.h"
// #include <thrust/iterator/constant_iterator.h>

namespace project_AntiRF {

// Warning: Only VID & VLABEL means VID=EID, VLABEL=ELABEL
// Warning: col_idx_ changed from size_t to VID, degree changed from size_t to
// VID
template <typename VID, typename VLABEL>
class Graph;

namespace dev {
template <typename VID, typename VLABEL>
class Graph {
 private:
  size_t vsize_;   // uninitialized
  size_t esize_;   // uninitialized
  VID max_degree;  // uninitialized
  Buffer<VLABEL> vlabels_;
  Buffer<VID> row_ptr_;
  Buffer<VID> col_idx_;
  Buffer<VID> odegs_;
  Buffer<VID> src_list_;
  Buffer<VID> dst_list_;

  template <typename _VID, typename _VLABEL>
  friend class project_AntiRF::Graph;

 public:
  // Graph(project_AntiRF::Graph& hg) { init(hg);}
  VID get_vnum() const { return vsize_; }
  VID get_enum() const { return esize_; }
  DEV_INLINE VID get_src(VID edge) const { return src_list_.data()[edge]; }
  DEV_INLINE VID get_dst(VID edge) const { return dst_list_.data()[edge]; }
  DEV_INLINE VID getOutDegree(VID src) {
    return col_idx_.data()[src + 1] - col_idx_.data()[src];
  }  // check
  DEV_INLINE VID edge_begin(VID src) { return col_idx_.data()[src]; }
  DEV_INLINE VID edge_end(VID src) { return col_idx_.data()[src + 1]; }
  DEV_INLINE VID get_edge_dst(VID idx) { return row_ptr_.data()[idx]; }
  // Test and dump COO
  DEV_INLINE void DumpCO() {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      printf("Dump COO: src_list size: %d, dst_list size: %d.\n",
             src_list_.size(), dst_list_.size());
      for (int i = 0; i < src_list_.size(); i++) {
        printf("%d ", src_list_.data()[i]);
      }
      printf("\n");
      for (int i = 0; i < dst_list_.size(); i++) {
        printf("%d ", dst_list_.data()[i]);
      }
    }
  }

  DEV_INLINE VID* getNeighbor(VID vid) const {
    // auto data_ = row_ptr_.data();
    // auto col_idx_.data()[vid];
    // VID* a = const_cast<VID*>(row_ptr_.data());
    // //VID* b = const_cast<VID*>(col_idx_.data()[vid]);
    // return row_ptr_.data() + col_idx_.data()[vid]* sizeof(VID);

    // col_idx_.data()[vid] is VID type, row_ptr_.data() is const VID* type. *
    // sizeof(VID)
    return const_cast<VID*>(row_ptr_.data()) + col_idx_.data()[vid];
  }
};
}  // namespace dev

template <typename VID, typename VLABEL>
class Graph {
 public:
  using device_t = dev::Graph<VID, VLABEL>;
  // TODO: To support multiple partition in vertex-cut manner.
  // To this end, we have to store the vertex mapping(original_id->local_id)
  // get the neighborlist START pointer
  VID* getNeighbor(VID vid) const {
    return const_cast<VID*>(row_ptr_.data()) + col_idx_.data()[vid];
  }
  VID edge_begin(VID src) { return col_idx_.data()[src]; }
  VID edge_end(VID src) { return col_idx_.data()[src + 1]; }

  VID get_src(VID idx) const { return src_list_[idx]; }
  VID get_dst(VID idx) const { return dst_list_[idx]; }

  VID get_vnum() const { return vsize_; }
  VID get_enum() const { return esize_; }
  VID getMaxDegree() { return max_degree_; }
  VID getOutDegree(VID src) {
    return col_idx_.data()[src + 1] - col_idx_.data()[src];
  }
  VID CalMaxDegree(std::vector<VID> out_degs) {
    auto maxPosition = max_element(out_degs.begin(), out_degs.end());
    return *maxPosition;
  }

  // USE_DAG on with orientation
  void orientation(bool NeedToLoadToDevice = true) {
    LOG(INFO) << "Orientation enabled, DAG generated.\n";
    double start = wtime();
    std::vector<VID> new_odegs_(vsize_, 0);
#pragma omp parallel for
    // Dump(std::cout);
    for (VID src = 0; src < vsize_; ++src) {
      VID* neighlist = getNeighbor(src);
      Buffer<VID> tmp(neighlist, getOutDegree(src));
      // LOG(INFO) << " size of neighlist: " << sizeof(neighlist);
      for (auto dst : tmp) {
        // LOG(INFO) << "i is " << i << ", dst is " << dst;
        if (odegs_[dst] > odegs_[src] ||
            (odegs_[dst] == odegs_[src] && dst > src)) {
          new_odegs_[src]++;
        }
      }
    }

    VID new_max_degree_ = CalMaxDegree(new_odegs_);
    LOG(INFO) << "Orientation Generating: New max degree is: "
              << new_max_degree_;
    // vector type: this.row_ptr_; this.col_idx_;
    std::vector<VID> new_row_ptr_;
    std::vector<VID> new_col_idx_;
    std::vector<VID> new_src_list_;
    new_col_idx_.resize(vsize_ + 1);
    parallel_prefix_sum<VID, VID>(new_odegs_,
                                  new_col_idx_.data());  // vector satisfied
    auto n_edges_ = new_col_idx_[vsize_];
    new_row_ptr_.resize(n_edges_);
    new_src_list_.resize(n_edges_);
#pragma omp parallel for
    for (VID src = 0; src < vsize_; ++src) {
      VID* neighlist = getNeighbor(src);
      Buffer<VID> tmp(neighlist, getOutDegree(src));
      auto begin = new_col_idx_[src];
      VID offset = 0;
      for (auto dst : tmp) {
        if (odegs_[dst] > odegs_[src] ||
            (odegs_[dst] == odegs_[src] && dst > src)) {
          new_row_ptr_[begin + offset] = dst;
          new_src_list_[begin + offset] = src;
          offset++;
        }
      }
    }
    //     for (VID src = 0; src < vsize_; ++src) {
    //   VID* neighlist = getNeighbor(src);
    //   auto begin = new_col_idx_[src];
    //   VID offset = 0;
    //   for(int i = 0; i < sizeof(neighlist) / sizeof(neighlist[0]); ++i) {
    //     auto dst = neighlist[i];
    //     if (odegs_[dst] > odegs_[src] || (odegs_[dst] == odegs_[src] && dst >
    //     src)) {
    //       new_row_ptr_[begin + offset] = dst;
    //       offset++;
    //     }
    //   }
    // }
    // Update graph info
    row_ptr_ = new_row_ptr_;
    col_idx_ = new_col_idx_;
    esize_ = n_edges_;
    max_degree_ = new_max_degree_;

    double end = wtime();
    LOG(INFO) << "Orientation Generating time: " << (end - start) << " seconds";
    // copy to GPU
    d_row_ptr_.resize(esize_);
    d_odegs_.resize(vsize_);
    d_col_idx_.resize(vsize_ + 1);

    // TODO: need consider symmetry break here!
    src_list_ = new_src_list_;
    dst_list_ = new_row_ptr_;
    if (NeedToLoadToDevice) {
      TODEV(thrust::raw_pointer_cast(d_src_list_.data()), src_list_.data(),
            sizeof(VID) * esize_);
      TODEV(thrust::raw_pointer_cast(d_dst_list_.data()), dst_list_.data(),
            sizeof(VID) * esize_);
      TODEV(thrust::raw_pointer_cast(d_row_ptr_.data()), row_ptr_.data(),
            sizeof(VID) * esize_);
      TODEV(thrust::raw_pointer_cast(d_odegs_.data()), odegs_.data(),
            sizeof(VID) * vsize_);
      TODEV(thrust::raw_pointer_cast(d_col_idx_.data()), col_idx_.data(),
            sizeof(VID) * (vsize_ + 1));  // size_ to VID
    }
  }
  //TODO: sort COO Graph
  void SortCSRGraph(bool NeedToLoadToDevice = true) {
    std::vector<int> index(vsize_);
    std::vector<int> r_index(vsize_);
    for (int i = 0; i < index.size(); i++)
      index[i] = i;
    std::stable_sort(index.begin(), index.end(), [&](int a, int b) {
      return getOutDegree(a) > getOutDegree(b);
    });

    std::vector<int> new_col_idx_(vsize_ + 1);
    std::vector<int> new_row_ptr_(esize_);
    std::vector<VID> new_odegs_(vsize_, 0);

    for (VID src = 0; src < vsize_; src++) {
      VID v = index[src];
      r_index[v] = src;
    }

    for (VID src = 0; src < vsize_; src++) {
      VID v = index[src];
      new_odegs_[src] = getOutDegree(v);
    }
    parallel_prefix_sum<VID, VID>(new_odegs_,
                                  new_col_idx_.data());  // vector satisfied
    for (VID src = 0; src < vsize_; src++) {
      VID v = index[src];
      VID* neighlist = getNeighbor(v);
      Buffer<VID> tmp(neighlist, getOutDegree(v));
      auto begin = new_col_idx_[src];
      VID offset = 0;
      for (auto dst : tmp) {
        new_row_ptr_[begin + offset] = r_index[dst];
        offset++;
      }
      std::sort(&new_row_ptr_[begin], &new_row_ptr_[begin + offset]);
    }

    col_idx_ = new_col_idx_;
    row_ptr_ = new_row_ptr_;
    odegs_ = new_odegs_;
    TODEV(thrust::raw_pointer_cast(d_row_ptr_.data()), row_ptr_.data(),
          sizeof(VID) * esize_);
    TODEV(thrust::raw_pointer_cast(d_odegs_.data()), odegs_.data(),
          sizeof(VID) * vsize_);
    TODEV(thrust::raw_pointer_cast(d_col_idx_.data()), col_idx_.data(),
          sizeof(VID) * (vsize_ + 1));  // size_ to VID
  }

  void Init(std::vector<VID> const& vids, std::vector<VLABEL> const& vlabels,
            std::vector<std::pair<VID, VID>> const& edges,
            bool use_label = false) {
    LOG(INFO) << "Initializing graph...";
    double start = wtime();

    vsize_ = vids.size();
    esize_ = edges.size();
    if (use_label)
      vlabels_ = std::move(vlabels);
    odegs_.resize(vsize_);
    col_idx_.resize(vsize_ + 1);
    row_ptr_.resize(esize_);

    src_list_.resize(esize_);
    dst_list_.resize(esize_);

    for (size_t i = 0; i < edges.size(); ++i) {
      odegs_[edges[i].first]++;
    }

    col_idx_[0] = 0;
    for (size_t i = 0; i < vsize_; ++i) {
      col_idx_[i + 1] = col_idx_[i] + odegs_[i];
      odegs_[i] = 0;
    }

    // directed edges
    for (size_t i = 0; i < esize_; ++i) {
      VID v0 = edges[i].first;
      VID v1 = edges[i].second;
      row_ptr_[col_idx_[v0] + odegs_[v0]] = v1;
      odegs_[v0]++;
    }

    double end = wtime();
    LOG(INFO) << "CSR transforming time: " << end - start << "s";
    LOG(INFO) << " -- vsize: " << vsize_ << " esize: " << esize_ << "\n";
    // calculate max degree
    max_degree_ = CalMaxDegree(odegs_);  // VID
    // LOG(INFO) << "Max degree cal from init. " << max_degree_ ;

    // generating COO
    // Note: May use vector<std::pair> instead.
    double start_coo = wtime();
    nnz = esize_;  // no sym_break, no ascend.
    for (size_t i = 0; i < esize_; ++i) {
      src_list_[i] = edges[i].first;
      dst_list_[i] = edges[i].second;
    }
    double end_coo = wtime();
    LOG(INFO) << "COO loading time: " << end_coo - start_coo << "s";

    // LOG(INFO) << "Dump COO: src_list_ size " << src_list_.size() << ",
    // dst_list_ size " << dst_list_.size(); for (int i = 0; i < esize_; ++i) {
    //   LOG(INFO) << src_list_.data()[i] << dst_list_.data()[i] << "\n";
    // }
    if (use_label)
      d_vlabels_.resize(vsize_);
    d_row_ptr_.resize(esize_);
    d_odegs_.resize(vsize_);
    d_col_idx_.resize(vsize_ + 1);
    d_src_list_.resize(esize_);
    d_dst_list_.resize(esize_);
    if (use_label)
      TODEV(thrust::raw_pointer_cast(d_vlabels_.data()), vlabels_.data(),
            sizeof(VLABEL) * vsize_);
    TODEV(thrust::raw_pointer_cast(d_row_ptr_.data()), row_ptr_.data(),
          sizeof(VID) * esize_);
    TODEV(thrust::raw_pointer_cast(d_odegs_.data()), odegs_.data(),
          sizeof(VID) * vsize_);
    TODEV(thrust::raw_pointer_cast(d_col_idx_.data()), col_idx_.data(),
          sizeof(VID) * (vsize_ + 1));  // size_ to VID
    TODEV(thrust::raw_pointer_cast(d_src_list_.data()), src_list_.data(),
          sizeof(VID) * esize_);
    TODEV(thrust::raw_pointer_cast(d_dst_list_.data()), dst_list_.data(),
          sizeof(VID) * esize_);
    // TOHOST(thrust::raw_pointer_cast(d_src_list_.data()), src_list_.data(),
    // sizeof(VID) * esize_);
    // TOHOST(thrust::raw_pointer_cast(d_dst_list_.data()), dst_list_.data(),
    // sizeof(VID) * esize_); LOG(INFO) << "After Copyback, now dump COO:
    // src_list_ size " << src_list_.size() << ", dst_list_ size " <<
    // dst_list_.size(); for (int i = 0; i < esize_; ++i) {
    //   LOG(INFO) << src_list_.data()[i] << dst_list_.data()[i] << "\n";
    // }
    // Checked ok.
  }

  device_t DeviceObject() const {  // DEV_HOST, now is HOST
    device_t dg;

    // dg.vlabels_ = Buffer<VLABEL>(d_vlabels_);
    dg.row_ptr_ = Buffer<VID>(d_row_ptr_);
    dg.odegs_ = Buffer<VID>(d_odegs_);
    dg.col_idx_ = Buffer<VID>(d_col_idx_);
    dg.src_list_ = Buffer<VID>(d_src_list_);
    dg.dst_list_ = Buffer<VID>(d_dst_list_);

    // dump test
    // std::vector<VID> sr_t;
    // std::vector<VID> ds_t;
    // sr_t.resize(esize_);
    // ds_t.resize(esize_);
    // TOHOST(thrust::raw_pointer_cast(d_src_list_.data()), sr_t.data(),
    // sizeof(VID) * esize_);
    // TOHOST(thrust::raw_pointer_cast(d_dst_list_.data()), ds_t.data(),
    // sizeof(VID) * esize_); LOG(INFO) << "In deviceobject, now dump COO: src
    // size " << sr_t.size() << ", dst size " << ds_t.size(); for (int i = 0; i
    // < esize_; ++i) {
    //   LOG(INFO) << sr_t[i] << " "<< ds_t[i] << "\n";
    // }
    return dg;
  }

  void Dump(std::ostream& out) {
    out << "vsize: " << vsize_ << " esize: " << esize_ << "\n";
    out << "labels: ";
    for (size_t i = 0; i < vsize_; ++i) {
      out << vlabels_[i] << " ";
    }
    out << "\n";
    out << "row_ptr: ";
    for (size_t i = 0; i < esize_; ++i) {
      out << row_ptr_[i] << " ";
    }
    out << "\n";
    out << "col_idx: ";
    for (size_t i = 0; i < vsize_ + 1; ++i) {
      out << col_idx_[i] << " ";
    }
    out << "\n";
  }

  void DumpCOO(std::ostream& out) {
    out << "vsize: " << vsize_ << " esize: " << esize_ << "\n";
    out << "labels: ";
    for (size_t i = 0; i < vsize_; ++i) {
      out << vlabels_[i] << " ";
    }
    out << "\n";
    out << "src_list: ";
    for (size_t i = 0; i < esize_; ++i) {
      out << src_list_[i] << " ";
    }
    out << "\n";
    out << "dst_list: ";
    for (size_t i = 0; i < esize_; ++i) {
      out << dst_list_[i] << " ";
    }
    out << "\n";
  }

 private:
  // Warning: NOT support device_id & n_gpu yet.
  size_t fid_;  // ?
  size_t vsize_;
  size_t esize_;
  VID max_degree_;
  std::vector<VLABEL> vlabels_;

  // int num_vertex_classes; // label classes count
  // may used by filter
  // std::vector<VID> labels_frequency_;
  // VID max_label_frequency_;
  // int max_label;
  // std::vector<nlf_map> nlf_;
  // std::vector<VID> sizes;
  // CSR
  std::vector<VID> row_ptr_;
  std::vector<VID> col_idx_;
  std::vector<VID> odegs_;  // <size_t>
  // add elabels_
  // COO
  VID nnz;
  std::vector<VID> src_list_;  // <size_t>
  std::vector<VID> dst_list_;  // <size_t>

  // Warning: More supported format may increase the storage.
  thrust::device_vector<VLABEL> d_vlabels_;
  thrust::device_vector<VID> d_row_ptr_;
  thrust::device_vector<VID> d_odegs_;
  thrust::device_vector<VID> d_col_idx_;
  thrust::device_vector<VID> d_src_list_;
  thrust::device_vector<VID> d_dst_list_;
};

}  // namespace project_AntiRF

#endif  // endif
