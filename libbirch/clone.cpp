/**
 * @file
 */
#include "libbirch/clone.hpp"

/* see memory.cpp, where some variables declared in clone.hpp are defined to
 * ensure correct order of initialization of all global variables */


#if ENABLE_LAZY_DEEP_CLONE
libbirch::EntryExitLock libbirch::freezeLock;
thread_local bool libbirch::freezeUnderway = false;
libbirch::EntryExitLock libbirch::finishLock;
thread_local bool libbirch::finishUnderway = false;
#endif
