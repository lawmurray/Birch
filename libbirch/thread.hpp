/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
/**
 * Number of threads.
 */
#ifdef _OPENMP
static const unsigned nthreads = omp_get_max_threads();
#else
static const unsigned nthreads = 1u;
#endif

/**
 * Thread id.
 */
#ifdef _OPENMP
static thread_local const unsigned tid = omp_get_thread_num();
#else
static const unsigned tid = 0u;
#endif
}
