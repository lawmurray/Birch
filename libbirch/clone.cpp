/**
 * @file
 */
#include "libbirch/clone.hpp"

template<class PointerType>
void bi::clone_start(PointerType& o, SharedPtr<Memo>& m) {
  #if DEEP_CLONE_STRATEGY == DEEP_CLONE_EAGER
  m = m->fork();
  clone_get(o, m);
  #elif DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZY
  clone_pull(o, m);
  m = m->fork();
  #elif DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZIER
  m = m->fork();
  #endif
}

template<class PointerType>
void bi::clone_continue(PointerType& o, SharedPtr<Memo>& m) {
  #if DEEP_CLONE_STRATEGY == DEEP_CLONE_EAGER
  m = cloneMemo;
  clone_get(o, m);
  #elif DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZY
  clone_pull(o, m);
  m = cloneMemo;
  clone_deep(o, m);
  #elif DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZIER
  if (!cloneMemo->hasAncestor(m.get())) {
    clone_pull(o, m);
  }
  m = cloneMemo;
  #endif
}

template<class PointerType>
void bi::clone_get(PointerType& o, SharedPtr<Memo>& m) {
  #if DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZIER
  clone_deep(o, m);
  #endif

  if (o.get() && o.get()->getMemo() != m.get()) {
    auto cloned = o.get()->clones.get(m.get());
    if (!cloned) {
      /* promote weak pointer to shared pointer for further null check */
      SharedPtr<Any> s = o;
      if (s) {
        #if DEEP_CLONE_STRATEGY == DEEP_CLONE_EAGER
        /* for an eager deep clone we must be cautious to avoid infinite
         * recursion; memory for the new object is allocated first and put
         * in the map in case of deeper pointers back to the same object; then
         * the new object is constructed; there is no risk of another thread
         * accessing the uninitialized memory as the deep clone is not
         * accessible to other threads until completion; the new object will
         * at least have completed the Counted() constructor to initialize
         * reference counts before any recursive clones occur */
        Any* alloc = static_cast<Any*>(allocate(s->getSize()));
        assert(alloc);
        Any* uninit = s->clones.uninitialized_put(m.get(), alloc);
        assert(uninit == alloc);  // should be no thread contention here
        auto prevMemo = cloneMemo;
        cloneMemo = m;
        cloned = s->clone(uninit);
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
        cloneMemo = m;
        s = s->clone();
        cloneMemo = prevMemo;
        cloned = o.get()->clones.put(m.get(), s.get());
        #endif
      }
    }
    o = cloned;
    m = cloned->getMemo();
  }
}

template<class PointerType>
void bi::clone_pull(PointerType& o, SharedPtr<Memo>& m) {
  #if DEEP_CLONE_STRATEGY == DEEP_CLONE_LAZIER
  clone_deep(o, m);
  #endif
  if (o && o.get()->getMemo() != m.get()) {
    auto object = o.get()->clones.get(m.get(), o.get());
    if (object != o.get()) {
      o = object;
      m = object->getMemo();
    }
  }
}

template<class PointerType>
void bi::clone_deep(PointerType& o, SharedPtr<Memo>& m) {
  SharedPtr<Memo> parent = m->parent;
  if (o && o.get()->getMemo() != m.get() && parent) {
    clone_deep(o, parent);
    o = o.get()->clones.get(parent.get(), o.get());
  }
}

template void bi::clone_start<bi::SharedPtr<bi::Any>>(SharedPtr<Any>& o, SharedPtr<Memo>& m);
template void bi::clone_continue<bi::SharedPtr<bi::Any>>(SharedPtr<Any>& o, SharedPtr<Memo>& m);
template void bi::clone_get<bi::SharedPtr<bi::Any>>(SharedPtr<Any>& o, SharedPtr<Memo>& m);
template void bi::clone_pull<bi::SharedPtr<bi::Any>>(SharedPtr<Any>& o, SharedPtr<Memo>& m);
template void bi::clone_deep<bi::SharedPtr<bi::Any>>(SharedPtr<Any>& o, SharedPtr<Memo>& m);

template void bi::clone_start<bi::WeakPtr<bi::Any>>(WeakPtr<Any>& o, SharedPtr<Memo>& m);
template void bi::clone_continue<bi::WeakPtr<bi::Any>>(WeakPtr<Any>& o, SharedPtr<Memo>& m);
template void bi::clone_get<bi::WeakPtr<bi::Any>>(WeakPtr<Any>& o, SharedPtr<Memo>& m);
template void bi::clone_pull<bi::WeakPtr<bi::Any>>(WeakPtr<Any>& o, SharedPtr<Memo>& m);
template void bi::clone_deep<bi::WeakPtr<bi::Any>>(WeakPtr<Any>& o, SharedPtr<Memo>& m);
