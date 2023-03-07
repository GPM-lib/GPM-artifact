#ifndef DS_BITMAP_H
#define DS_BITMAP_H

#include <iomanip>
#include <iostream>

#include "src/utils/buffer.h"
#include "src/utils/cuda_utils.h"
#include "src/utils/utils.h"

namespace project_AntiRF {

template <typename word_t, int sft>
class Bitmap;

namespace dev {
template <typename word_t, int sft>
class Bitmap {
 public:
  int get_size() { return n_words; }

  // if bits_[v]==0 return true else false
  DEV_INLINE bool query(int v) {
    word_t mask_word = LoadWord(v);
    word_t mask_bit = Loc(v);
    if (!(mask_bit & mask_word))
      return true;
    return false;
  }

  // maybe cache_friendly
  DEV_INLINE bool query_bytes(int v) {
    char mask_word = ToBytes()[v >> 3];
    char mask_bit = 1 << (v & 7);
    if (!(mask_bit & mask_word))
      return true;
    else
      return false;
  }

  // mark the bits_ of v
  // WARNING: no consistency guarantee, so hope for the best.
  DEV_INLINE void mark(int v) {
    word_t mask_word = LoadWord(v);  // [A]
    word_t mask_bit = Loc(v);
    if (!(mask_bit & mask_word)) {
      do {
        mask_word |= mask_bit;
        StoreWord(v, mask_word);  // others may commit changes after [A]
        mask_word = LoadWord(v);
      } while (!(mask_bit & mask_word));
    }
  }

  // mark the bit of v, and return true if the bit==0;
  // WARNING: no consistency guarantee, so hope for the best.
  DEV_INLINE bool query_and_mark(int v) {
    word_t mask_word = LoadWord(v);  // [A]
    word_t mask_bit = Loc(v);
    if (!(mask_bit & mask_word)) {
      do {
        mask_word |= mask_bit;
        StoreWord(v, mask_word);  // others may commit changes after [A]
        mask_word = LoadWord(v);
      } while (!(mask_bit & mask_word));
      return true;
    }
    return false;
  }

  // mark the bit of v in atomic manner.
  // it guarantees the consistency, with higher overhead.
  DEV_INLINE bool query_and_mark_aotmic(int v) {
    int mask_bit = LocAsInt(v);
    int x = atomicOr(LoadWordPosAsInt(v), mask_bit);
    if (!(x & mask_bit))
      return true;
    return false;
  }

 private:
  DEV_INLINE char* ToBytes() { return (char*) (void*) dg_bits_.data(); }

  DEV_INLINE int SizeOfBytes() { return n_words << (sft - 3); }

  DEV_INLINE word_t LoadWord(int v) { return dg_bits_[v >> sft]; }

  DEV_INLINE word_t LoadWordPos(int v) { return &dg_bits_[v >> sft]; }

  DEV_INLINE void StoreWord(int v, word_t word) { dg_bits_[v >> sft] = word; }

  DEV_INLINE int Loc(int v) { return (1 << (v & this->MASK)); }

  // required by query_and_mark_aotmic
  DEV_INLINE int* LoadWordPosAsInt(int v) {
    int* ptr = (int*) dg_bits_.data();
    return &ptr[v >> 5];
  }

  DEV_INLINE int LocAsInt(int v) { return (1 << (v & 31)); }

  size_t const MASK = (1 << sft) - 1;
  Buffer<word_t> dg_bits_;
  int n_words;
  template <typename _word_t, int _sft>
  friend class project_AntiRF::Bitmap;
};
}  // namespace dev

// Bitmap<char, 3>;
// Bitmap<int, 5>;
template <typename word_t, int sft>
class Bitmap {
 public:
  using device_t = dev::Bitmap<word_t, sft>;
  Bitmap() {}
  explicit Bitmap(size_t n_size) : n_size_(n_size) { Resize(n_size); }

  void Resize(int n_size) {
    size_t n_words = (n_size + (1 << sft) - 1) >> sft;
    bits_.resize(n_words);
  }

  size_t Size() { return n_size_; }

  device_t DeviceObject() const {
    device_t d;
    d.dg_bits_ = Buffer<word_t>(bits_);
    d.n_words = bits_.size();
    return d;
  }

  void Reset() { thrust::fill(bits_.begin(), bits_.end(), 0); }

  void MarkAll() {
    thrust::fill(bits_.begin(), bits_.end(), ~(word_t) (0));
  }  // all become 0xf

  void MarkOne(int victim) {
    int loc = (victim >> sft);
    word_t word = 1 << (victim & this->MASK);
    bits_[loc] = word;
  }

  void Dump(std::ostream& out) {
    out << "bits_ size: " << bits_.size() << std::endl;
    out << "bits_ (hex): ";
    for (int i = bits_.size() - 1; i >= 0; --i) {
      out << std::hex << std::setw(sizeof(word_t) * 2) << std::setfill('0')
          << +bits_[i] << " ";
    }
    out << std::endl;
  }

 private:
  size_t n_size_;
  thrust::device_vector<word_t> bits_;
  size_t const MASK = (1 << sft) - 1;
};

}  // namespace project_AntiRF

#endif  // DS_BITMAP_
