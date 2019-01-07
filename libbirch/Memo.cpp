/**
 * @file
 */
#include "libbirch/Memo.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

bi::Memo::Memo() :
    isForked(false) {
  //
}

bi::Memo::Memo(Memo* parent, const bool isForwarding) :
    cloneParent(isForwarding ? nullptr : parent),
    forwardParent(isForwarding ? parent : nullptr),
    isForked(false) {
  //
}

bi::Memo::~Memo() {
  //
}

bool bi::Memo::hasAncestor(Memo* memo) const {
  SharedPtr<Memo> parent = getParent();
  return parent && (parent == memo || parent->hasAncestor(memo));
}

bi::Memo* bi::Memo::fork() {
  #if USE_LAZY_DEEP_CLONE
  isForked = true;
  #endif
  return create(this, false);
}

void bi::Memo::clean() {
  /* for current use cases, just clean the current memo, not parents */
  //auto parent = getParent();
  //if (parent) {
  //  parent->clean();
  //}
  clones.clean();
}

bi::Memo* bi::Memo::forwardGet() {
  #if USE_LAZY_DEEP_CLONE
  ///@todo make this thread safe
  if (forwardChild) {
    return forwardChild->forwardGet();
  } else if (isForked) {
    forwardChild = create(this, true);
    clean();
    return forwardChild.get();
  } else {
    return this;
  }
  #else
  return this;
  #endif
}

bi::Memo* bi::Memo::forwardPull() {
  #if USE_LAZY_DEEP_CLONE
  return forwardChild ? forwardChild->forwardPull() : this;
  #else
  return this;
  #endif
}

bi::SharedPtr<bi::Memo> bi::Memo::getParent() const {
  return cloneParent ? cloneParent : SharedPtr<Memo>(forwardParent);
}

bi::Any* bi::Memo::get(Any* o) {
  if (!o || o->getContext() == this) {
    return o;
  } else {
    Any* result = clones.get(o);
    if (!result) {
      #if USE_LAZY_DEEP_CLONE
      /* for a lazy deep clone there is no risk of infinite recursion, but
       * there may be thread contention if two threads access the same object
       * and both trigger a lazy clone simultaneously; in this case multiple
       * new objects may be made but only one thread can be successful in
       * inserting an object into the map; a shared pointer is used to
       * destroy any additional objects */
      SwapClone swapClone(true);
      SwapContext swapContext(this);
      SharedPtr<Any> cloned = o->clone();
      // ^ use shared to clean up if beaten by another thread
      result = clones.put(o, cloned.get());
      #else
      /* for an eager deep clone we must be cautious to avoid infinite
       * recursion; memory for the new object is allocated first and put
       * in the map in case of deeper pointers back to the same object; then
       * the new object is constructed; there is no risk of another thread
       * accessing the uninitialized memory as the deep clone is not
       * accessible to other threads until completion; the new object will
       * at least have completed the Counted() constructor to initialize
       * reference counts before any recursive clones occur */
      Any* alloc = static_cast<Any*>(allocate(o->getSize()));
      assert(alloc);
      Any* uninit = m->clones.uninitialized_put(o, alloc);
      assert(uninit == alloc);  // should be no thread contention here
      SwapClone swapClone(true);
      SwapContext swapContext(this);
      result = o->clone(uninit);
      assert(result == uninit);  // clone should be in the allocation
      o->incMemo();  // uninitialized_put(), so responsible for ref counts
      result->incShared();
      #endif
    }
    return result;
  }
}

bi::Any* bi::Memo::pull(Any* o) {
  if (!o || o->getContext() == this) {
    return o;
  } else {
    return clones.get(o, o);
  }
}

bi::Any* bi::Memo::deep(Any* o) {
  if (!o || o->getContext() == this) {
    return o;
  } else {
    Any* result = clones.get(o);
    if (!result) {
      auto parent = getParent();
      if (parent) {
        result = parent->deep(o);
        if (result != o) {
          result = clones.get(result, result);
        }
      } else {
        result = o;
      }
      result = clones.put(o, result);
    }
    return result;
  }
}
