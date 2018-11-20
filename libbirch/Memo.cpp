/**
 * @file
 */
#include "libbirch/Memo.hpp"

bi::Memo::Memo(Memo* parent) :
    parent(parent) {
  //
}

bi::Memo::~Memo() {
  //
}

bool bi::Memo::hasAncestor(Memo* memo) const {
  return parent == memo || (parent && parent->hasAncestor(memo));
}

bi::Any* bi::Memo::get(Any* o) {
  if (o->getMemo() == this) {
    return o;
  } else {
    auto cloned = clones.get(o);
    if (cloned) {
      return cloned;
    } else {
      #if DEEP_CLONE_STRATEGY == DEEP_CLONE_EAGER
      /* for an eager deep clone we must be cautious to avoid infinite
       * recursion; memory for the new object is allocated first and put
       * in the map in case of deeper pointers back to the same object; then
       * the new object is constructed; there is no risk of another thread
       * accessing the uninitialized memory as the deep clone is not
       * accessible to other threads until completion; the new object will
       * at least have completed the Counted() constructor to initialize
       * reference counts before any recursive clones occur */
      Any* alloc = static_cast<Any*>(allocate(o->getSize()));
      Any* uninit = clones.uninitialized_put(o, alloc);
      assert(uninit == alloc);   // should be no thread contention here
      auto prevMemo = cloneMemo;
      cloneMemo = this;
      cloned = o->clone(uninit);
      cloneMemo = prevMemo;
      assert(cloned == uninit);  // clone should be in the allocation
      cloned->incShared();
      return cloned;
      #else
      /* for a lazy deep clone there is no risk of infinite recursion, but
       * there may be thread contention if two threads access the same object
       * and both trigger a lazy clone simultaneously; in this case multiple
       * new objects may be made but only one thread can be successful in
       * inserting an object into the map; a shared pointer is used to
       * destroy any additional objects */
      auto prevMemo = cloneMemo;
      cloneMemo = this;
      SharedPtr<Any> cloned = o->clone();
      cloneMemo = prevMemo;
      return clones.put(o, cloned.get());
      #endif
    }
  }
}

bi::Any* bi::Memo::pull(Any* o) {
  if (o->getMemo() == this) {
    return o;
  } else {
    return clones.get(o, o);
  }
}

bi::Any* bi::Memo::deep(Any* o) {
  if (o->getMemo() == this || !parent) {
    return o;
  } else {
    return parent->deep(o);
  }
}
