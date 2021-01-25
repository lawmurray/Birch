/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/thread.hpp"

namespace libbirch {
class Any;
template<class T, class F> class Array;
template<class T> class Shared;

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

/**
 * Deep copy an object.
 */
Any* copy(Any* o);

}
