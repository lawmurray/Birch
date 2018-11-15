/**
 * @file
 */
#include "libbirch/Any.hpp"

#include "libbirch/global.hpp"
#include "libbirch/memory.hpp"
#include "libbirch/Enter.hpp"

bi::Any::Any() {
  //
}

bi::Any::Any(const Any& o) {
  //
}

bi::Any::~Any() {
  //
}

bi::Any* bi::Any::clone() const {
  return make_object<Any>(*this);
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
      assert(memo);
      Enter enter(memo);
      SharedPtr<Any> cloned(this->clone());
      cloned->setMemo(memo);
      // ^ shared pointer used so as to destroy object if another thread
      //   clones in the meantime
      return clones.put(memo, cloned.get());
    }
  }
}

bi::Any* bi::Any::map(Memo* memo) {
  if (!memo || getMemo() == memo) {
    return this;
  } else {
    return clones.get(memo, this);
  }
}

bi::Any* bi::Any::deepPull(Memo* memo) {
  if (!memo || getMemo() == memo) {
    return this;
  } else {
    auto pulled = deepPull(memo->getParent());
    return clones.get(memo, pulled);
  }
}
