/**
 * @file
 */
#include "numbirch/common/random.hpp"

namespace numbirch {
thread_local std::mt19937 rng32;
thread_local std::mt19937_64 rng64;
}
