#ifndef UTILS_FLAGS_H
#define UTILS_FLAGS_H

#include <gflags/gflags.h>

namespace project_AntiRF {

#ifndef GFLAGS_NAMESPACE
namespace flags = GFLAGS_NAMESPACE;
#else
namespace flags = gflags;
#endif

};  // namespace project_AntiRF

#endif  // UTILS_FLAGS_H
