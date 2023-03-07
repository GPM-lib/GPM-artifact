#ifndef UTILS_MACROS_H
#define UTILS_MACROS_H

#include <cstdlib>
#include <iostream>

namespace project_AntiRF {

struct Empty {
  char placeholder;
};

std::istream& operator>>(std::istream& in, Empty& e) {
  in >> e.placeholder;
  return in;
}

std::ostream& operator<<(std::ostream& out, Empty& e) {
  out << e.placeholder;
  return out;
}

enum algorithms { TC, SM, CF, MC, FSM };
enum modes { v_centric, e_centric };
typedef unsigned long long AccType;

}  // namespace project_AntiRF

#endif  // UTILS_MACROS_H
