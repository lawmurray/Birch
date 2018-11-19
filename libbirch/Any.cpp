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
      #if DEEP_CLONE_STRATEGY == DEEP_CLONE_EAGER
      /* for an eager deep clone we must be cautious to avoid infinite
       * recursion; memory for the new object is allocated first and put
       * in the map in case of deeper pointers back to the same object; then
       * the new object is constructed; there is no risk of another thread
       * accessing the uninitialized memory as the deep clone is not
       * accessible to other threads until completion; the new object will
       * at least have completed the Counted() constructor to initialize
       * reference counts before any recursive clones occur */
      Any* alloc = static_cast<Any*>(allocate(this->size));
      Any* uninit = clones.uninitialized_put(memo, alloc);
      assert(uninit == alloc);   // should be no thread contention here
      auto prevMemo = cloneMemo;
      cloneMemo = memo;
      cloned = this->clone(uninit);
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
      cloneMemo = memo;
      SharedPtr<Any> cloned = this->clone();
      cloneMemo = prevMemo;
      return clones.put(memo, cloned.get());
      #endif
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
