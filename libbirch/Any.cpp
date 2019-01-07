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
  //
}

bi::Memo* bi::Any::getContext() {
  return context.get();
}
