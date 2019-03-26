/**
 * @file
 */
#pragma once

#ifdef _OPENMP
#include <omp.h>
#endif

namespace libbirch {
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
