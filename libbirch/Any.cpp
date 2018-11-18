/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/clone.hpp"
#include "libbirch/memory.hpp"

bi::Any::Any() :
    memo(cloneMemo) {
  //
}

bi::Any::Any(const Any& o) :
    memo(cloneMemo) {
  //
}

bi::Any::~Any() {
  //
}

bi::Memo* bi::Any::getMemo() {
  return memo.get();
}

bi::Any* bi::Any::get(Memo* memo) {
  if (!memo || getMemo() == memo) {
    return this;
  } else {
    auto cloned = clones.get(memo);
    if (cloned) {
      return cloned;
    } else {
      cloneMemo = memo;
      SharedPtr<Any> cloned = this->clone();
      // ^ shared pointer used so as to destroy object if another thread
      //   clones in the meantime
      cloneMemo = nullptr;
      return clones.put(memo, cloned.get());
    }
  }
}

bi::Any* bi::Any::pull(Memo* memo) {
  if (!memo || getMemo() == memo) {
    return this;
  } else {
    return clones.get(memo, this);
  }
}

bi::Any* bi::Any::deepGet(Memo* memo) {
  return deepPull(memo)->get(memo);
}

bi::Any* bi::Any::deepPull(Memo* memo) {
  if (!memo || getMemo() == memo) {
    return this;
  } else {
    return deepPull(memo->getParent())->pull(memo);
  }
}
