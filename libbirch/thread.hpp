/**
 * @file
 */
#pragma once

#include "libbirch/external.hpp"
#include "libbirch/EntryExitLock.hpp"

namespace libbirch {
class Label;

/**
 * The root context.
 */
extern Label* rootContext;

/**
 * Global freeze lock.
 */
extern EntryExitLock freezeLock;

/**
 * Global finish lock.
 */
extern EntryExitLock finishLock;

}
