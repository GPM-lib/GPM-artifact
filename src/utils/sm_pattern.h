#ifndef UTILS_SM_PATTERN_H
#define UTILS_SM_PATTERN_H

#include "src/utils/utils.h"
#define MAX_PATTERN_SIZE 8

namespace project_AntiRF {
// motif-counting patterns
static int num_possible_patterns[] = {
    0,      1, 1,
    2,       // size 3
    6,       // size 4
    21,      // size 5
    112,     // size 6
    853,     // size 7
    11117,   // size 8
    261080,  // size 9
};

enum Labelling { UNLABELLED, LABELLED, PARTIALLY_LABELLED, DISCOVER_LABELS };

class Pattern {
 private:
  std::string name_;
  bool has_label;

 public:
  Pattern() : Pattern("") {}
  Pattern(std::string name) : name_(name) {}
  // Note: file input pattern analysis skipped.
  ~Pattern() {}
  // pattern pre-defined
  bool is_rectangle() const { return name_ == "rectangle"; }
  bool is_pentagon() const { return name_ == "pentagon"; }
  bool is_house() const { return name_ == "house"; }

  std::string get_name() const { return name_; }
};

}  // namespace project_AntiRF

#endif  // endif