/**
 * @file
 */
#pragma once

/**
 * @def DISABLE_POOL
 *
 * Set to 1 to use standard malloc/realloc/free, and disable OpenMP,
 * for memory leak checks with valgrind.
 */
#define DISABLE_POOL 0

#include "libbirch/Pool.hpp"

#include <type_traits>
#include <random>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <cassert>
#include <vector>

/**
 * @def bi_assert
 *
 * If debugging is enabled, check an assertion and abort on fail.
 */
#ifndef NDEBUG
#define bi_assert(cond) if (!(cond)) bi::abort()
#else
#define bi_assert(cond)
#endif

/**
 * @def bi_assert_msg
 *
 * If debugging is enabled, check an assertion and abort with a message on
 * fail.
 */
#ifndef NDEBUG
#define bi_assert_msg(cond, msg) \
  if (!(cond)) { \
    std::stringstream buf; \
    buf << msg; \
    bi::abort(buf.str()); \
  }
#else
#define bi_assert_msg(cond, msg)
#endif

/**
 * @def bi_function
 *
 * Push a new frame onto the stack trace.
 */
#ifndef NDEBUG
#define bi_function(func, file, n) StackFunction function(func, file, n)
#else
#define bi_function(func, file, n)
#endif

/**
 * @def bi_line
 *
 * Update the line number of the top frame on the stack trace.
 */
#ifndef NDEBUG
#define bi_line(n) stacktrace.back().line = n
#else
#define bi_line(n)
#endif

namespace bi {
class World;
class Any;
template<class T> class SharedCOW;
template<class T> class WeakCOW;

/**
 * Stack frame.
 */
struct StackFrame {
  const char* func;
  const char* file;
  int line;
};

/**
 * Temporary type for pushing functions onto the stack trace.
 */
struct StackFunction {
  StackFunction(const char* func, const char* file, const int line);
  ~StackFunction();
};

/**
 * The world of the currently running fiber.
 */
extern World* fiberWorld;
#pragma omp threadprivate(fiberWorld)

/**
 * Flag set when an object is being cloned.
 */
extern bool fiberClone;
#pragma omp threadprivate(fiberClone)

/**
 * Constant to indicate a mutable value. Zero is convenient here, as it
 * enables multiplication to convolve multiple values.
 *
 * @ingroup libbirch
 */
static constexpr int64_t mutable_value = 0;

/**
 * Stack trace.
 */
extern std::vector<StackFrame> stacktrace;
#pragma omp threadprivate(stacktrace)

/**
 * Report unknown program option and abort.
 */
void unknown_option(const std::string& name);

/**
 * Print stack trace and abort.
 */
void abort();

/**
 * Print stack trace and abort with message.
 */
void abort(const std::string& msg);

inline bi::StackFunction::StackFunction(const char* func, const char* file,
    const int line) {
  stacktrace.push_back({ func, file, line });
}

inline bi::StackFunction::~StackFunction() {
  stacktrace.pop_back();
}

/**
 * The super type of type @p T. Specialised in forward declarations of
 * classes.
 */
template<class T>
struct super_type {
  using type = Any;
};

template<class T>
struct super_type<const T> {
  using type = const typename super_type<T>::type;
};

/**
 * Does type @p T hs an assignment operator for type @p U?
 */
template<class T, class U>
struct has_assignment {
  static const bool value =
      has_assignment<typename super_type<T>::type,U>::value;
};
template<class U>
struct has_assignment<Any,U> {
  static const bool value = false;
};

/**
 * Does type @p T have a conversion operator for type @p U?
 */
template<class T, class U>
struct has_conversion {
  static const bool value =
      has_conversion<typename super_type<T>::type,U>::value;
};
template<class U>
struct has_conversion<Any,U> {
  static const bool value = false;
};

#if !DISABLE_POOL
/**
 * Buffer for heap allocations.
 */
extern char* buffer;

/**
 * Allocation pools.
 */
extern bi::Pool pool[];
#endif

/**
 * For an allocation size, determine the index of the pool to which it
 * belongs.
 *
 * @param n Number of bytes.
 *
 * @return Pool index.
 *
 * Pool sizes are multiples of 8 bytes up to 64 bytes, and powers of two
 * thereafter.
 */
inline int bin(const size_t n) {
#ifdef HAVE___BUILTIN_CLZLL
  return (n <= 64ull) ? ((unsigned)n - 1u) >> 3u : 65 - __builtin_clzll(n - 1ull);
#else
  if (n <= 64ull) {
    return ((unsigned)n - 1u) >> 3u;
  } else {
    unsigned ret = 1u;
    while (((n - 1ull) >> ret) > 0ull) {
      ++ret;
    }
    return (int)ret + 1;
  }
#endif
}

/**
 * For an allocation size, determine the index of the pool to which it
 * belongs.
 *
 * @param n Number of bytes.
 *
 * @return Pool index.
 *
 * Pool sizes are multiples of 8 bytes up to 64 bytes, and powers of two
 * thereafter.
 */
inline int bin(const unsigned n) {
#ifdef HAVE___BUILTIN_CLZ
  return (n <= 64u) ? (n - 1u) >> 3u : 33 - __builtin_clz(n - 1u);
#else
  if (n <= 64u) {
    return (n - 1u) >> 3u;
  } else {
    unsigned ret = 1u;
    while (((n - 1u) >> ret) > 0u) {
      ++ret;
    }
    return (int)ret + 1;
  }
#endif
}

/**
 * For an allocation size, determine the index of the pool to which it
 * belongs.
 *
 * @tparam n Number of bytes.
 *
 * @return Pool index.
 *
 * Pool sizes are multiples of 8 bytes up to 64 bytes, and powers of two
 * thereafter.
 */
template<unsigned n>
inline int bin() {
#ifdef HAVE___BUILTIN_CLZLL
  return (n <= 64u) ? (n - 1u) >> 3u : 8*sizeof(unsigned) - __builtin_clz(n - 1u) + 1;
#else
  if (n <= 64u) {
    return (n - 1u) >> 3u;
  } else {
    unsigned ret = 1u;
    while (((n - 1u) >> ret) > 0u) {
      ++ret;
    }
    return (int)ret + 1;
  }
#endif
}

/**
 * Determine the size for a given bin.
 */
inline size_t unbin(const int i) {
  return (i <= 7) ? (i + 1) << 3 : (1ull << (i - 1ull));
}

/**
 * Allocate memory from heap.
 *
 * @param n Number of bytes.
 *
 * @return Pointer to the allocated memory.
 */
void* allocate(const size_t n);

/**
 * Allocate memory from heap.
 *
 * @tparam n Number of bytes.
 *
 * @return Pointer to the allocated memory.

 * This implementation, where the size is given by a static 32-bit
 * integer, is typically slightly faster than the 64-bit integer
 * version.
 */
template<unsigned n>
void* allocate() {
#if DISABLE_POOL
  return std::malloc(n);
#else
  void* ptr = nullptr;
  if (n > 0u) {
    int i = bin<n>();     // determine which pool
    ptr = pool[i].pop();  // attempt to reuse from this pool
    if (!ptr) {           // otherwise allocate new
      size_t m = unbin(i);
      #pragma omp atomic capture
      {
        ptr = buffer;
        buffer += m;
      }
    }
    assert(ptr);
  }
  return ptr;
#endif
}

/**
 * Deallocate memory from the heap, previously allocated with
 * allocate() or reallocate().
 *
 * @param ptr Pointer to the allocated memory.
 * @param n Number of bytes.
 */
void deallocate(void* ptr, const size_t n);

/**
 * Deallocate memory from the heap, previously allocated with
 * allocate() or reallocate().
 *
 * @param ptr Pointer to the allocated memory.
 * @param n Number of bytes.
 *
 * This implementation, where the size is given by a 32-bit integer,
 * is typically slightly faster than the 64-bit integer version.
 */
void deallocate(void* ptr, const unsigned n);

/**
 * Reallocate memory from heap.
 *
 * @param ptr1 Pointer to the allocated memory.
 * @param n1 Number of bytes in current allocated memory.
 * @param n2 Number of bytes in newly allocated memory.
 *
 * @return Pointer to the newly allocated memory.
 */
void* reallocate(void* ptr1, const size_t n1, const size_t n2);

/**
 * Construct an object with placement new using memory obtained from
 * allocate();
 *
 * @tparam T Class type.
 * @param Args... Constructor argument types.
 *
 * @param args Construct arguments.
 *
 * @return Pointer to the object.
 */
template<class T, class ... Args>
inline T* construct(Args ... args) {
  return new (allocate<sizeof(T)>()) T(args...);
}
}
