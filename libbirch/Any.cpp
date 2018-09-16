/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/global.hpp"
#include "libbirch/memory.hpp"

bi::Any::Any() :
    memo(fiberMemo) {
  //
}

bi::Any::Any(const Any& o) :
    memo(fiberMemo) {
  //
}

bi::Any::~Any() {
  //
}

bi::Any* bi::Any::clone() const {
  return bi::construct<Any>(*this);
}

void bi::Any::destroy() {
  this->size = sizeof(*this);
  this->~Any();
}

bi::Memo* bi::Any::getMemo() {
  return memo;
}
