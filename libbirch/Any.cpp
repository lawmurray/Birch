/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/memory.hpp"

bi::Any::Any() :
    memo(cloneMemo->forwardGet()) {
  //
}

bi::Any::Any(const Any& o) :
    memo(cloneMemo->forwardGet()) {
  //
}

bi::Any::~Any() {
  //
}

bi::Memo* bi::Any::getMemo() {
  return memo.get();
}
