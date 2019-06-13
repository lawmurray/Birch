/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"

namespace libbirch {
/**
 * Number of threads.
 */
#pragma omp declare target
#ifdef _OPENMP
static const unsigned nthreads = omp_get_max_threads();
#else
static const unsigned nthreads = 1u;
#endif
#pragma omp end declare target

/**
 * Thread id.
 */
#pragma omp declare target
#ifdef _OPENMP
static /*thread_local*/ const unsigned tid = omp_get_thread_num();
#else
static const unsigned tid = 0u;
#endif
#pragma omp end declare target
}
