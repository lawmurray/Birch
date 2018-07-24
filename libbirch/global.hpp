/**
 * @file
 */
#pragma once

#include <type_traits>
#include <random>
#include <memory>
#include <cstdint>
#include <cstddef>
#include <list>
#include <cassert>
#include <vector>
#include <stack>

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
#define bi_line(n) stacktrace.front().line = n
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
extern std::list<StackFrame> stacktrace;
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
  stacktrace.push_front( { func, file, line });
}

inline bi::StackFunction::~StackFunction() {
  stacktrace.pop_front();
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

/**
 * Small object allocation buffer. This is used for objects < 64 bytes in
 * size, where allocations are not necessarily aligned to cache lines.
 */
extern char* smallBuffer;

/**
 * Large object allocation buffer. This is used for objects >= 64 bytes in
 * size, in which case they are also a power of two, ensuring that all
 * allocations are aligned to cache lines.
 */
extern char* largeBuffer;

/**
 * Allocation pool.
 */
extern std::stack<void*,std::vector<void*>> pool[];

/**
 * Determine in which bin an allocation of size @p n belongs. Return the
 * index of the bin and the size of allocations in that bin (which will
 * be greater than or equal to @p n).
 */
inline int bin(const size_t n) {
#ifdef HAVE___BUILTIN_CLZLL
  return (n <= 64ull) ? ((unsigned)n - 1u) >> 3u : 8*sizeof(long long) - __builtin_clzll(n - 1ull) + 2;
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

inline int bin(const unsigned n) {
#ifdef HAVE___BUILTIN_CLZLL
  return (n <= 64u) ? (n - 1u) >> 3u : 8*sizeof(long long) - __builtin_clz(n - 1u) + 2;
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

template<unsigned n>
inline int bin() {
#ifdef HAVE___BUILTIN_CLZLL
  return (n <= 64u) ? (n - 1u) >> 3u : 8*sizeof(unsigned) - __builtin_clz(n - 1u) + 2;
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
  return (i <= 7) ? (i + 1) << 3 : (1ull << (i - 2ull));
}

/**
 * Allocate with static size.
 */
template<unsigned n>
void* allocate() {
  void* ptr = nullptr;
  if (n > 0u) {
    /* bin the allocation */
    int i = bin<n>();

    /* reuse allocation in the pool, or create a new one */
    auto& p = pool[i];
    if (p.empty()) {
      size_t m = unbin(i);
      if (i <= 7) {
#pragma omp atomic capture
        {
          ptr = smallBuffer;
          smallBuffer += m;
        }
      } else {
#pragma omp atomic capture
        {
          ptr = largeBuffer;
          largeBuffer += m;
        }
      }
    } else {
      ptr = p.top();
      p.pop();
    }
    assert(ptr);
  }
  return ptr;
}

void* allocate(const size_t n);
void* reallocate(void* ptr1, const size_t n1, const size_t n2);
void deallocate(void* ptr, const size_t n);
void deallocate(void* ptr, const unsigned n);

template<class T, class ... Args>
inline T* construct(Args ... args) {
  return new (allocate<sizeof(T)>()) T(args...);
}
}
