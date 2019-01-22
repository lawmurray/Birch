/**
 * @file
 */
#include "libbirch/thread.hpp"

#ifdef _OPENMP
unsigned bi::nthreads = omp_get_max_threads();
#else
unsigned bi::nthreads = 1u;
#endif

#ifdef _OPENMP
unsigned bi::tid = omp_get_thread_num();
#else
unsigned bi::tid = 0u;
#endif
