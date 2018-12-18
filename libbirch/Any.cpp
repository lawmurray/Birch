/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/memory.hpp"

bi::Any::Any() :
    context(top_context()->forwardGet()) {
  //
}

bi::Any::Any(const Any& o) :
    Counted(o),
    context(top_context()) {
  //
}

bi::Any::~Any() {
  //
}

bi::Memo* bi::Any::getContext() {
  return context.get();
}
