/**
 * @file
 */
#include "libbirch/thread.hpp"

#if ENABLE_LAZY_DEEP_CLONE
libbirch::EntryExitLock libbirch::freezeLock;
libbirch::EntryExitLock libbirch::finishLock;
#endif
