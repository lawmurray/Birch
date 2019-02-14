/**
 * @file
 */
#pragma once

#include "libbirch/thread.hpp"
#include "libbirch/Pool.hpp"

#include <atomic>

namespace bi {
/**
 * Buffer for heap allocations.
 */
extern std::atomic<char*> buffer;

/**
 * Start of heap (for debugging purposes).
 */
extern char* bufferStart;

/**
 * Size of heap (for debugging purposes).
 */
extern size_t bufferSize;

/**
 * Allocation pools.
 */
extern Pool* pool;

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
  int result;
#ifdef HAVE___BUILTIN_CLZLL
  result = (n <= 64ull) ? ((unsigned)n - 1u) >> 3u : 65 - __builtin_clzll(n - 1ull);
#else
  if (n <= 64ull) {
    result = ((unsigned)n - 1u) >> 3u;
  } else {
    unsigned ret = 1u;
    while (((n - 1ull) >> ret) > 0ull) {
      ++ret;
    }
    result = (int)ret + 1;
  }
#endif
  assert(0 <= result && result <= 63);
  return result;
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
  int result;
#ifdef HAVE___BUILTIN_CLZ
  result = (n <= 64u) ? (n - 1u) >> 3u : 33 - __builtin_clz(n - 1u);
#else
  if (n <= 64u) {
    result = (n - 1u) >> 3u;
  } else {
    unsigned ret = 1u;
    while (((n - 1u) >> ret) > 0u) {
      ++ret;
    }
    result = (int)ret + 1;
  }
#endif
  assert(0 <= result && result <= 63);
  return result;
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
  int result;
#ifdef HAVE___BUILTIN_CLZLL
  result = (n <= 64u) ? (n - 1u) >> 3u : 8*sizeof(unsigned) - __builtin_clz(n - 1u) + 1;
#else
  if (n <= 64u) {
    result = (n - 1u) >> 3u;
  } else {
    unsigned ret = 1u;
    while (((n - 1u) >> ret) > 0u) {
      ++ret;
    }
    result = (int)ret + 1;
  }
#endif
  assert(0 <= result && result <= 63);
  return result;
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
  static_assert(n > 0, "cannot make zero length allocation");
#if !ENABLE_MEMORY_POOL
  return std::malloc(n);
#else
  int i = bin<n>();     // determine which pool
  auto ptr = pool[64 * tid + i].pop();  // attempt to reuse from this pool
  if (!ptr) {           // otherwise allocate new
    unsigned m = unbin(i);
    unsigned r = (m < 64u) ? 64u : m;
    // ^ minimum allocation 64 bytes to maintain alignment
    ptr = buffer.fetch_add(r, std::memory_order_relaxed);
    if (m < 64u) {
      /* add extra bytes as a separate allocation to the pool for
       * reuse another time */
      pool[64 * tid + bin(64u - m)].push((char*)ptr + m);
    }
  }
  assert(ptr);
  return ptr;
#endif
}

/**
 * Deallocate memory from the heap, previously allocated with
 * allocate() or reallocate().
 *
 * @param ptr Pointer to the allocated memory.
 * @param n Number of bytes.
 * @param tid Id of thread that originally allocated.
 */
void deallocate(void* ptr, const size_t n, const unsigned tid);

/**
 * Deallocate memory from the heap, previously allocated with
 * allocate() or reallocate().
 *
 * @param ptr Pointer to the allocated memory.
 * @param n Number of bytes.
 * @param tid Id of thread that originally allocated.
 *
 * This implementation, where the size is given by a 32-bit integer,
 * is typically slightly faster than the 64-bit integer version.
 */
void deallocate(void* ptr, const unsigned n, const unsigned tid);

/**
 * Reallocate memory from heap.
 *
 * @param ptr1 Pointer to the allocated memory.
 * @param n1 Number of bytes in current allocated memory.
 * @param tid1 Id of thread that originally allocated.
 * @param n2 Number of bytes for newly allocated memory.
 *
 * @return Pointer to the newly allocated memory.
 */
void* reallocate(void* ptr1, const size_t n1, const unsigned tid1,
    const size_t n2);
}
