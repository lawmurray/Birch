/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/memory.hpp"

bi::Any::Any() :
    context(cloneMemo->forwardGet()) {
  //
}

bi::Any::Any(const Any& o) :
    Counted(o),
    context(cloneMemo) {
  //
}

bi::Any::~Any() {
  //
}

bi::Memo* bi::Any::getContext() {
  return context.get();
}
