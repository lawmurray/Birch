/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/EntryExitLock.hpp"

namespace libbirch {
class LazyLabel;

#if ENABLE_LAZY_DEEP_CLONE
/**
 * The root context.
 */
extern LazyLabel* rootContext;

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
