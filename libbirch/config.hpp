/**
 * @file
 */
#pragma once

#include <algorithm>
#include <numeric>
#include <utility>
#include <functional>
#include <initializer_list>
#include <memory>
#include <atomic>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <getopt.h>
#include <dlfcn.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/**
 * @def DISABLE_POOL
 *
 * Set to 1 to use standard malloc/realloc/free instead of pooled allocator.
 *
 * For memory leak checks with valgrind, set this to 1, and may also need to
 * disable OpenMP when compiling.
 */
#define DISABLE_POOL 0

/**
 * @def DEEP_CLONE
 *
 * Set to:
 *
 * @li 0 to use an eager deep clone,
 * @li 1 to use a lazy deep clone with eager map,
 * @li 2 to use a lazy deep clone with eager map only when necessary.
 */
#define DEEP_CLONE 1
