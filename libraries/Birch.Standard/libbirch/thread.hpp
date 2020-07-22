/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {

/**
 * Get the maximum number of threads.
 *
 * @ingroup libbirch
 */
inline int get_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

/**
 * Get the current thread's number.
 *
 * @ingroup libbirch
 */
inline int get_thread_num() {
#ifdef _OPENMP
  return omp_get_thread_num();
#else
  return 0;
#endif
}

}
