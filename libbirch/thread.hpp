/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/Context.hpp"
#include "libbirch/EntryExitLock.hpp"

namespace libbirch {
/**
 * The current context to which to assign newly created objects in the
 * thread.
 */
thread_local extern Context* currentContext;

/**
 * Is a clone operation currently underway in the thread?
 *
 * @note A thread-local variable is used rather than a function
 * argument to facilitate use of copy constructors when cloning objects,
 * particularly important when using types from e.g. the STL where we can't
 * modify the copy constructor or add new constructors.
 */
thread_local extern bool cloneUnderway;

#if ENABLE_LAZY_DEEP_CLONE
/**
 * Global freeze lock.
 */
extern EntryExitLock freezeLock;

/**
 * Global finish lock.
 */
extern EntryExitLock finishLock;
#endif
}
