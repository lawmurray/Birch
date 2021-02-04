/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/internal.hpp"
#include "libbirch/thread.hpp"

namespace libbirch {
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
 * Query or toggle the biconnected-copy flag.
 * 
 * @param toggle If true, the flag is toggled (false to true, or true to
 * false).
 * 
 * @return New state of the flag.
 * 
 * This is used to distinguish the two contexts in which the copy constructor
 * of Shared<T> is called: one where a copy of the next biconnected component
 * should be triggered, the other while that is ongoing and the encounter of
 * further bridges should not trigger further copies.
 * 
 * The initial state of the flag is false.
 */
bool biconnected_copy(const bool toggle = false);

}
