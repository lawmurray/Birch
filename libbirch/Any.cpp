/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/clone.hpp"
#include "libbirch/memory.hpp"

bi::Any::Any() :
    memo(globalMemo) {
  //
}

bi::Any::Any(const Any& o) :
    memo(globalMemo) {
  //
}

bi::Any::~Any() {
  //
}

bi::Memo* bi::Any::getMemo() {
  return memo.get();
}
