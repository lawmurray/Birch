/**
 * @file
 */
#include "libbirch/Memo.hpp"

#include "libbirch/Any.hpp"
#include "libbirch/Enter.hpp"
#include "libbirch/Clone.hpp"

bi::Memo::Memo() :
    parent(nullptr),
    internal(false) {
  //
}

bi::Memo::Memo(int) :
    parent(nullptr),
    internal(false) {
  incShared();
}

bi::Memo::Memo(SharedPtr<Memo> parent) :
    parent(parent),
    internal(false) {
  parent->internal = true;
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
  if (!o || this == o->getMemo()) {
    return o;
  } else {
    auto cloned = clones.get(o);
    if (cloned) {
      return cloned;
    } else {
      /* shouldn't be in a position to be cloning objects in interior
       * memos */
      assert(!internal);

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
  if (!o || this == o->getMemo()) {
    return o;
  } else {
    return clones.get(o, o);
  }
}

bi::Any* bi::Memo::deepPull(Any* o) {
  if (!o || this == o->getMemo()) {
    return o;
  } else {
    auto pulled = parent->deepPull(o);
    return clones.get(pulled, pulled);
  }
}
