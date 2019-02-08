/**
 * @file
 */
#pragma once

#include "libbirch/config.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace bi {
/**
 * Number of threads.
 */
extern unsigned nthreads;

/**
 * Thread id.
 */
extern unsigned tid;
#pragma omp threadprivate(tid)
}
