/**
 * @file
 */
#include "libbirch/thread.hpp"

/**
 * Get the root context.
 */
static libbirch::Context* root() {
  static libbirch::SharedPtr<libbirch::Context> context(new libbirch::Context());
  return context.get();
}

thread_local libbirch::Context* libbirch::currentContext(root());
thread_local bool libbirch::cloneUnderway = false;

#if ENABLE_LAZY_DEEP_CLONE
libbirch::EntryExitLock libbirch::freezeLock;
libbirch::EntryExitLock libbirch::finishLock;
#endif
