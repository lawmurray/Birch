/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Context.hpp"
#include "libbirch/EntryExitLock.hpp"

namespace libbirch {
/**
 * The current context to which to assign newly created objects in the
 * thread.
 */
thread_local extern Context* currentContext;

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

#if ENABLE_LAZY_DEEP_CLONE
/**
 * Global freeze lock.
 */
extern EntryExitLock freezeLock;

/**
 * Global finish lock.
 */
extern EntryExitLock finishLock;
#endif
}
