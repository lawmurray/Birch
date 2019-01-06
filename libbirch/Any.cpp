/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/memory.hpp"

bi::Any::Any() :
    context(currentContext) {
  //
}

bi::Any::Any(const Any& o) :
    Counted(o),
    context(currentContext) {
  //
}

bi::Any::~Any() {
  #if USE_LAZY_DEEP_CLONE && USE_LAZY_DEEP_CLONE_FORWARD_CLEAN
  memos.destroy(this);
  #endif
}

bi::Memo* bi::Any::getContext() {
  return context.get();
}

#if USE_LAZY_DEEP_CLONE && USE_LAZY_DEEP_CLONE_FORWARD_CLEAN
void bi::Any::recordMemo(Memo* o) {
  memos.put(o);
}
#endif
