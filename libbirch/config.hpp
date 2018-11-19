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
#include <cassert>
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
#ifndef DISABLE_POOL
#define DISABLE_POOL 0
#endif

/**
 * @def DEEP_CLONE
 *
 * Set to:
 *
 * @li 1 to use an eager deep clone,
 * @li 2 to use a lazy deep clone with eager map,
 * @li 3 to use a lazy deep clone with eager map only when necessary.
 */
#define DEEP_CLONE_EAGER 1
#define DEEP_CLONE_LAZY 2
#define DEEP_CLONE_LAZIER 3

#ifndef DEEP_CLONE_STRATEGY
#define DEEP_CLONE_STRATEGY DEEP_CLONE_LAZY
#endif

/**
 * @def STANDARD_CREATE_FUNCTION
 *
 * Defines the standard @c create() member function required of objects.
 */
#define STANDARD_CREATE_FUNCTION \
  template<class... Args> \
  static class_type* create(Args... args) { \
    return emplace(allocate<sizeof(class_type)>(), args...); \
  }

/**
 * @def STANDARD_EMPLACE_FUNCTION
 *
 * Defines the standard @c emplace() member function required of objects.
 */
#define STANDARD_EMPLACE_FUNCTION \
  template<class... Args> \
  static class_type* emplace(void* ptr, Args... args) { \
    auto o = new (ptr) class_type(args...); \
    o->size = sizeof(class_type); \
    return o; \
  }

/**
 * @def STANDARD_CLONE_FUNCTION
 *
 * Defines the standard @c clone() member function required of objects.
 */
#define STANDARD_CLONE_FUNCTION \
  virtual class_type* clone() const { \
    return emplace(allocate<sizeof(class_type)>(), *this); \
  }

/**
 * @def STANDARD_DESTROY_FUNCTION
 *
 * Defines the standard @c destroy() member function required of objects.
 */
#define STANDARD_DESTROY_FUNCTION \
  virtual void destroy() override { \
    this->~class_type(); \
  }

/**
 * Constant to indicate a mutable value. Zero is convenient here, as it
 * enables multiplication to convolve multiple values.
 *
 * @ingroup libbirch
 */
static constexpr int64_t mutable_value = 0;
