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

bi::Memo* bi::Memo::forward() {
  if (child) {
    return child->forward();
  } else {
    return this;
  }
}

bi::Memo* bi::Memo::fork() {
  assert(!child);
  child = create(this);
  if (globalMemo.get() == this) {
    globalMemo = child;
  }
  return create(this);
}

std::tuple<bi::Any*,bi::Memo*> bi::Memo::get(Any* o) {
  #if DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZIER
  o = deep(o);
  #endif
  if (!o || o->getMemo() == this) {
    return std::make_tuple(o, this);
  } else {
    SharedPtr<Any> cloned = clones.get(o);
    if (!cloned) {
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
      #else
      /* for a lazy deep clone there is no risk of infinite recursion, but
       * there may be thread contention if two threads access the same object
       * and both trigger a lazy clone simultaneously; in this case multiple
       * new objects may be made but only one thread can be successful in
       * inserting an object into the map; a shared pointer is used to
       * destroy any additional objects */
      auto prevMemo = cloneMemo;
      cloneMemo = this;
      cloned = o->clone();
      cloneMemo = prevMemo;
      cloned = clones.put(o, cloned.get());
      #endif
    }
    return std::make_tuple(cloned.get(), cloned->getMemo());
  }
}

std::tuple<bi::Any*,bi::Memo*> bi::Memo::pull(Any* o) {
  #if DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZIER
  o = deep(o);
  #endif
  Any* object = o;
  Memo* memo = this;

  if (o && o->getMemo() != this) {
    object = clones.get(o, o);
    if (object != o) {
      memo = object->getMemo();
    }
  }
  return std::make_tuple(object, memo);
}

bi::Any* bi::Memo::deep(Any* o) {
  if (o->getMemo() == this || !parent) {
    return o;
  } else {
    auto pulled = parent->deep(o);
    return parent->clones.get(pulled, pulled);
  }
}

std::tuple<bi::Any*,bi::Memo*> bi::Memo::copy(Any* o) {
  Any* object = o;
  Memo* memo = this;
  #if DEEP_CLONE_STRATEGY == DEEP_CLONE_EAGER
  std::tie(object, memo) = memo->pull(object);
  std::tie(object, memo) = cloneMemo->get(object);
  #elif DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZY
  std::tie(object, memo) = memo->pull(object);
  object = cloneMemo->deep(object);
  #elif DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZIER
  if (!cloneMemo->hasAncestor(memo)) {
    std::tie(object, memo) = memo->pull(object);
  }
  memo = cloneMemo;
  #endif
  return std::make_tuple(object, memo);
}

std::tuple<bi::Any*,bi::Memo*> bi::Memo::clone(Any* o) {
  Any* object;
  Memo* memo;
  #if DEEP_CLONE_STRATEGY == DEEP_CLONE_EAGER
  std::tie(object, memo) = pull(o);
  SharedPtr<Memo> child(memo->fork());  // so that destroyed on return
  object = child->get(object);
  memo = globalMemo;
  #elif DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZY
  std::tie(object, memo) = pull(o);
  memo = memo->fork();
  #elif DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZIER
  object = o;
  memo = fork();
  #endif
  return std::make_tuple(object, memo);
}
