/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/EntryExitLock.hpp"

namespace libbirch {
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
