/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/global.hpp"
#include "libbirch/memory.hpp"

bi::Any::Any() {
  //
}

bi::Any::Any(const Any& o) {
  //
}

bi::Any::~Any() {
  //
}

bi::Any* bi::Any::clone(Memo* memo) const {
  return clone_object(this, memo);
}

void bi::Any::destroy() {
  this->size = sizeof(*this);
  this->~Any();
}

bi::Memo* bi::Any::getMemo() {
  return memo.get();
}

void bi::Any::setMemo(Memo* memo) {
  this->memo = memo;
}

bi::Any* bi::Any::get(Memo* memo) {
  if (!memo || getMemo() == memo) {
    return this;
  } else {
    auto cloned = clones.get(memo);
    if (cloned) {
      return cloned;
    } else {
      SharedPtr<Any> cloned(this->clone(memo));
      cloned->setMemo(memo);
      // ^ shared pointer used so as to destroy object if another thread
      //   clones in the meantime
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
