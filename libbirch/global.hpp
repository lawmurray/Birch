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
 * @def BIRCH_POOL_ALLOCATOR
 *
 * Set to enable the pool allocator. Disable if checking for memory
 * leaks with valgrind.
 */
#define BIRCH_POOL_ALLOCATOR 1

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

/**
 * Flag set when an object is being cloned.
 */
extern bool fiberClone;

/**
 * Constant to indicate a mutable value. Zero is convenient here, as it
 * enables multiplication to convolve multiple values.
 *
 * @ingroup libbirch
 */
static constexpr int64_t mutable_value = 0;

/**
 * Allocation pool.
 */
extern std::stack<void*,std::vector<void*>> pool[];

/**
 * Stack trace.
 */
extern std::list<StackFrame> stacktrace;

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
 * Determine in which bin an allocation of size @p n belongs. Return the
 * index of the bin and the size of allocations in that bin (which will
 * be greater than or equal to @p n).
 */
inline int bin(const size_t n) {
#if __has_builtin(__builtin_clzll)
  return sizeof(unsigned long long)*8 - __builtin_clzll(n - 1);
#else
  int ret = 1;
  while (((n - 1) >> ret) > 0) {
    ++ret;
  }
  return ret;
#endif
}

inline void* allocate(const size_t n) {
#if BIRCH_POOL_ALLOCATOR
  void* ptr = nullptr;
  if (n > 0) {
    /* bin the allocation */
    int i = bin(n);

    /* reuse allocation in the pool, or create a new one */
    auto& p = pool[i];
    if (p.empty()) {
      ptr = std::malloc(1 << i);
    } else {
      ptr = p.top();
      p.pop();
    }
    assert(ptr);
  }
  return ptr;
#else
  return std::malloc(n);
#endif
}

inline void* reallocate(void* ptr1, const size_t n1, const size_t n2) {
#if BIRCH_POOL_ALLOCATOR
  void* ptr2 = nullptr;

  /* bin the current allocation */
  int i1 = bin(n1);

  /* bin the new allocation */
  int i2 = bin(n2);

  if (n1 > 0 && i1 == i2) {
    /* current allocation is correct size, reuse */
    ptr2 = ptr1;
  } else {
    if (n2 > 0) {
      /* reuse allocation in the pool, or create a new one */
      auto& p = pool[i2];
      if (p.empty()) {
        ptr2 = std::malloc(1 << i2);
      } else {
        ptr2 = p.top();
        p.pop();
      }
      assert(ptr2);

      /* copy over contents */
      std::memcpy(ptr2, ptr1, n1);

      /* return the previous allocation to its pool */
      if (n1 > 0) {
        pool[i1].push(ptr1);
      }
    }
  }
  return ptr2;
#else
  return std::realloc(ptr1, n2);
#endif
}

inline void deallocate(void* ptr, const size_t n) {
#if BIRCH_POOL_ALLOCATOR
  if (n > 0) {
    assert(ptr);

    /* bin the allocation */
    int i = bin(n);

    /* return the allocation to its pool */
    pool[i].push(ptr);
  }
#else
  std::free(ptr);
#endif
}

template<class T, class ... Args>
inline T* construct(Args ... args) {
  return new (bi::allocate(sizeof(T))) T(args...);
}

}
