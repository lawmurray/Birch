/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

/**
 * @internal
 * 
 * The absolute maximum number of threads supported. This is the maximum
 * regardless of other settings, as it is used to allocate static arrays.
 */
#define LIBBIRCH_MAX_THREADS 4096

namespace libbirch {
/**
 * Get the maximum number of threads.
 *
 * @ingroup libbirch
 */
inline int get_max_threads() {
#ifdef _OPENMP
  return std::min(omp_get_max_threads(), LIBBIRCH_MAX_THREADS);
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
