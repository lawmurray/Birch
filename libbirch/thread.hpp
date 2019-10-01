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
