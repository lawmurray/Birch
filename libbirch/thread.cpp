/**
 * @file
 */
#include "libbirch/thread.hpp"

#include "libbirch/LazyLabel.hpp"

#if ENABLE_LAZY_DEEP_CLONE
static libbirch::LazyLabel* root() {
  static libbirch::SharedPtr<libbirch::LazyLabel> context(new libbirch::Label());
  return context.get();
}

libbirch::LazyLabel* libbirch::rootContext = root();
libbirch::EntryExitLock libbirch::freezeLock;
libbirch::EntryExitLock libbirch::finishLock;
#endif
