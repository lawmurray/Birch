/**
 * @file
 */
#include "libbirch/Memo.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/Clone.hpp"

bi::Memo::Memo() :
    parent(nullptr) {
  //
}

bi::Memo::Memo(int) :
    parent(nullptr) {
  incShared();
}

bi::Memo::Memo(const SharedPtr<Memo>& parent) :
    parent(parent) {
  //
}

bi::Memo::~Memo() {
  //
}

void bi::Memo::destroy() {
  size = sizeof(*this);
  this->~Memo();
}

bool bi::Memo::hasAncestor(Memo* memo) const {
  return this == memo || (parent && parent->hasAncestor(memo));
}

bi::Any* bi::Memo::get(Any* o) {
  if (this == o->getMemo()) {
    return o;
  } else {
    auto cloned = clones.get(o);
    if (cloned) {
      return cloned;
    } else {
      Enter enter(this);
      Clone clone;
      SharedPtr<Any> cloned(o->clone());
      // ^ shared pointer used so as to destroy object if another thread
      //   clones in the meantime
      return clones.put(o, cloned.get());
    }
  }
}

bi::Any* bi::Memo::pull(Any* o) {
  if (this == o->getMemo()) {
    return o;
  } else {
    auto cloned = clones.get(o);
    if (cloned) {
      return cloned;
    } else {
      return o;
    }
  }
}
