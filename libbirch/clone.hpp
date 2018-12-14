/**
 * @file
 */
#pragma once

#include "libbirch/ContextPtr.hpp"

namespace bi {
/**
 * Start a deep clone.
 */
template<class PointerType>
void clone_start(PointerType& o, ContextPtr& m);

/**
 * Continue a deep clone.
 */
template<class PointerType>
void clone_continue(PointerType& o, ContextPtr& m);

/**
 * Shallow mapping of an object that may not yet have been cloned,
 * cloning it if necessary.
 */
template<class PointerType>
void clone_get(PointerType& o, ContextPtr& m);

/**
 * Shallow mapping of an object that may not yet have been cloned,
 * without cloning it. This can be used as an optimization for read-only
 * access.
 */
template<class PointerType>
void clone_pull(PointerType& o, ContextPtr& m);

/**
 * Deep mapping of an object through ancestor memos up to the current memo,
 * witout any cloning; get() or pull() should be called on the result to
 * map through this memo.
 *
 * @param o The source object.
 *
 * @return The mapped object.
 */
template<class PointerType>
void clone_deep(PointerType& o, const SharedPtr<Memo>& m);

}
