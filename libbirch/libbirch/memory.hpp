/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/internal.hpp"

namespace libbirch {
/**
 * Register an object with the cycle collector as the possible root of a
 * cycle. This corresponds to the `PossibleRoot()` operation in @ref Bacon2001
 * "Bacon & Rajan (2001)".
 */
void register_possible_root(Any* o);

/**
 * Attempt to deregister an object with the cycle collector. The operation is
 * optional and only performed if it can be done efficiently, such as if the
 * object is the most-recently registered.
 */
void deregister_possible_root(Any* o);

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

/**
 * Collect a biconnected component, outside of regular garbage collection.
 * 
 * @param o Head of the biconnected component.
 */
void biconnected_collect(Any* o);

}
