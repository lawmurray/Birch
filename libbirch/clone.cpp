/**
 * @file
 */
#include "libbirch/clone.hpp"

template<class PointerType>
void bi::clone_start(PointerType& o, ContextPtr& m) {
  #if USE_LAZY_DEEP_CLONE
  clone_pull(o, m);
  m = m->fork();
  #else
  m = m->fork();
  clone_get(o, m);
  #endif
}

template<class PointerType>
void bi::clone_continue(PointerType& o, ContextPtr& m) {
  #if USE_LAZY_DEEP_CLONE
  clone_pull(o, m);
  m = contexts.back().get();
  clone_deep(o, m->getParent());
  #else
  m = contexts.back();
  clone_get(o, m);
  #endif
}

template<class PointerType>
void bi::clone_get(PointerType& o, ContextPtr& m) {
  assert(o);
  if (o.get()->getContext() != m.get()) {
    auto cloned = m->clones.get(o.get());
    if (!cloned) {
      /* promote weak pointer to shared pointer for further null check */
      SharedPtr<Any> s = o;
      if (s) {
        #if USE_LAZY_DEEP_CLONE
        /* for a lazy deep clone there is no risk of infinite recursion, but
         * there may be thread contention if two threads access the same object
         * and both trigger a lazy clone simultaneously; in this case multiple
         * new objects may be made but only one thread can be successful in
         * inserting an object into the map; a shared pointer is used to
         * destroy any additional objects */
        auto prevUnderway = cloneUnderway;
        contexts.push_back(m.get());
        cloneUnderway = true;
        s = s->clone();
        cloned = m->clones.put(o.get(), s.get());
        contexts.pop_back();
        contexts.back() = contexts.back()->forwardPull();
        cloneUnderway = prevUnderway;
        #if USE_LAZY_DEEP_CLONE_FORWARD_CLEAN
        if (cloned == s.get()) {  // weren't beaten by another thread
          o.get()->recordClone(cloned);
        }
        #endif

        #else
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
        Any* uninit = m->clones.uninitialized_put(s.get(), alloc);
        assert(uninit == alloc);  // should be no thread contention here
        auto prevUnderway = cloneUnderway;
        contexts.push_back(m);
        cloneUnderway = true;
        cloned = s->clone(uninit);
        contexts.pop_back();
        cloneUnderway = prevUnderway;
        assert(cloned == uninit);  // clone should be in the allocation
        m->incWeak();  // uninitialized_put(), so responsible for ref counts
        cloned->incShared();
        #endif
      }
    }
    o = cloned;
  }
}

template<class PointerType>
void bi::clone_pull(PointerType& o, ContextPtr& m) {
  assert(o);
  if (o.get()->getContext() != m.get()) {
    o = m->clones.get(o.get(), o.get());
  }
}

template<class PointerType>
void bi::clone_deep(PointerType& o, const SharedPtr<Memo>& m) {
  assert(o);
  if (m && o.get()->getContext() != m.get()) {
    Any* from = o.get();
    Any* to = m->clones.get(from);
    if (to) {
      o = to;
    } else {
      clone_deep(o, m->getParent());
      o = m->clones.get(o.get(), o.get());
      to = m->clones.put(from, o.get());
    }
  }
}

template void bi::clone_start<bi::SharedPtr<bi::Any>>(SharedPtr<Any>& o, ContextPtr& m);
template void bi::clone_continue<bi::SharedPtr<bi::Any>>(SharedPtr<Any>& o, ContextPtr& m);
template void bi::clone_get<bi::SharedPtr<bi::Any>>(SharedPtr<Any>& o, ContextPtr& m);
template void bi::clone_pull<bi::SharedPtr<bi::Any>>(SharedPtr<Any>& o, ContextPtr& m);
template void bi::clone_deep<bi::SharedPtr<bi::Any>>(SharedPtr<Any>& o, const SharedPtr<Memo>& m);

template void bi::clone_start<bi::WeakPtr<bi::Any>>(WeakPtr<Any>& o, ContextPtr& m);
template void bi::clone_continue<bi::WeakPtr<bi::Any>>(WeakPtr<Any>& o, ContextPtr& m);
template void bi::clone_get<bi::WeakPtr<bi::Any>>(WeakPtr<Any>& o, ContextPtr& m);
template void bi::clone_pull<bi::WeakPtr<bi::Any>>(WeakPtr<Any>& o, ContextPtr& m);
template void bi::clone_deep<bi::WeakPtr<bi::Any>>(WeakPtr<Any>& o, const SharedPtr<Memo>& m);
