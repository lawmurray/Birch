/**
 * @file
 */
#include "libbirch/thread.hpp"

/**
 * Get the root context.
 */
static libbirch::Label* root() {
  static libbirch::SharedPtr<libbirch::Label> context(new libbirch::Label());
  return context.get();
}

thread_local libbirch::Label* libbirch::currentContext(root());
thread_local bool libbirch::cloneUnderway = false;

#if ENABLE_LAZY_DEEP_CLONE
libbirch::EntryExitLock libbirch::freezeLock;
libbirch::EntryExitLock libbirch::finishLock;
#endif
