/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/thread.hpp"
#include "libbirch/Atomic.hpp"
#include "libbirch/Pool.hpp"
#include "libbirch/ExitBarrierLock.hpp"

namespace libbirch {
class Any;
class Label;

/**
 * Lock for sharing finish operations. Finish operations may intersect on the
 * graph of reachable objects, and so workshare. This lock creates a barrier
 * on exit to ensure that all reachable objects are visited despite this
 * sharing.
 */
extern ExitBarrierLock finish_lock;

/**
 * Lock for sharing freeze operations.
 *
 * @seealso finish_lock
 */
extern ExitBarrierLock freeze_lock;

/**
 * Get the root label.
 */
Label*& root();

/**
 * Get the heap.
 */
Atomic<char*>& heap();

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
  assert(n > 0ull);
  int result = 0;
  #ifdef HAVE___BUILTIN_CLZLL
  if (n > 64ull) {
    /* __builtin_clzll undefined for zero argument */
    result = 64 - __builtin_clzll((n - 1ull) >> 6ull);
  }
  #else
  while (((n - 1ull) >> (6 + result)) > 0) {
    ++result;
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
 * Pool sizes are powers of two, with a minimum allocation of 64 bytes to
 * avoid false sharing.
 */
inline int bin(const unsigned n) {
  assert(n > 0);
  int result = 0;
  #ifdef HAVE___BUILTIN_CLZ
  if (n > 64u) {
    /* __builtin_clz undefined for zero argument */
    result = 32 - __builtin_clz((n - 1u) >> 6u);
  }
  #else
  while (((n - 1u) >> (6 + result)) > 0) {
    ++result;
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
 * Pool sizes are powers of two, with a minimum allocation of 64 bytes to
 * avoid false sharing.
 */
template<unsigned n>
inline int bin() {
  assert(n > 0);
  int result = 0;
  #ifdef HAVE___BUILTIN_CLZ
  if (n > 64u) {
    /* __builtin_clz undefined for zero argument */
    result = 32 - __builtin_clz((n - 1u) >> 6u);
  }
  #else
  while (((n - 1u) >> (6 + result)) > 0) {
    ++result;
  }
  #endif
  assert(0 <= result && result <= 63);
  return result;
}

/**
 * Determine the size for a given bin.
 */
inline size_t unbin(const int i) {
  return 64ull << i;
}

/**
 * Get the `i`th pool.
 */
Pool& pool(const int i);

/**
 * Allocate memory from heap.
 *
 * @param n Number of bytes.
 *
 * @return Pointer to the allocated memory.
 */
void* allocate(const size_t n);

/**
 * Deallocate memory from the heap, previously allocated with
 * allocate() or reallocate().
 *
 * @param ptr Pointer to the allocated memory.
 * @param n Number of bytes.
 * @param tid Id of thread that originally allocated.
 */
void deallocate(void* ptr, const size_t n, const int tid);

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
void deallocate(void* ptr, const unsigned n, const int tid);

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
void* reallocate(void* ptr1, const size_t n1, const int tid1,
    const size_t n2);

/**
 * Register an object with the cycle collector as the possible root of a
 * cycle. This corresponds to the `PossibleRoot()` operation in @ref Bacon2001
 * "Bacon & Rajan (2001)".
 */
void register_possible_root(Any* o);

/**
 * Register an object with the cycle collector as unreachable.
 */
void register_unreachable(Any* o);

/**
 * Run the cycle collector.
 */
void collect();

/**
 * Performs some maintenance operations on the current thread's set of
 * registered possible roots.
 *
 * @param o The object that called this operation, and that is not a possible
 * root.
 *
 * Specifically, from the back of the vector of possible roots, this removes
 * any pointers to objects that are (no longer) possible roots, either because
 * they are flagged as such, or because they match `o`. Working from the back
 * is a reasonable heuristic, especially for pointers on the stack, which
 * a destroyed in the reverse order in which they are created.
 */
void trim(Any* o);

}
