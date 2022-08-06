/**
 * @file
 */
#pragma once

#include "membirch/external.hpp"
#include "membirch/internal.hpp"

namespace membirch {
/**
 * @internal
 * 
 * Register an object with the cycle collector as the possible root of a
 * cycle. This corresponds to the `PossibleRoot()` operation in @ref Bacon2001
 * "Bacon & Rajan (2001)".
 */
void register_possible_root(Any* o);

/**
 * @internal
 * 
 * Attempt to deregister an object with the cycle collector. The operation is
 * optional and only performed if it can be done efficiently, such as if the
 * object is the most-recently registered.
 */
void deregister_possible_root(Any* o);

/**
 * @internal
 * 
 * Is the object in this thread's possible root list? Useful for debugging
 * purposes.
 */
bool contains_possible_root(Any* o);

/**
 * @internal
 * 
 * Register an object with the cycle collector as unreachable.
 */
void register_unreachable(Any* o);

/**
 * Run the cycle collector. This must be called from outside of a parallel
 * region.
 */
void collect();

/**
 * @internal
 * 
 * Is the copy flag set?
 * 
 * This is used to distinguish the two contexts in which the copy constructor
 * of Shared<T> is called: one where a copy of the next biconnected component
 * should be triggered, the other while that is ongoing and the encounter of
 * further bridges should not trigger further copies.
 * 
 * The initial state of the flag is false.
 */
bool in_copy();

/**
 * @internal
 * 
 * Set the copy flag.
 */
void set_copy();

/**
 * @internal
 * 
 * Unset the copy flag.
 */
void unset_copy();

/**
 * @internal
 * 
 * Collect a biconnected component, outside of regular garbage collection.
 * 
 * @param o Head of the biconnected component.
 */
void biconnected_collect(Any* o);

}
