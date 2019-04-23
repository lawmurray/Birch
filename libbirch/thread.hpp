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
thread_local extern unsigned nthreads;

/**
 * Thread id.
 */
thread_local extern unsigned tid;
}
