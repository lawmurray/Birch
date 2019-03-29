/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#include "libbirch/LazyMemo.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

libbirch::LazyMemo::LazyMemo() :
    parent(nullptr),
    forward(nullptr),
    gen(0u) {
  //
}

libbirch::LazyMemo::LazyMemo(LazyMemo* parent) :
    parent(parent),
    forward(nullptr),
    gen(parent->gen + 1u) {
  assert(parent);
}

libbirch::LazyMemo::~LazyMemo() {
  LazyMemo* forward = this->forward.load(std::memory_order_relaxed);
  if (forward) {
    forward->decShared();
  }
}


bool libbirch::LazyMemo::hasAncestor(LazyMemo* memo) {
  if (gen <= memo->gen) {
    return false;
  } else if (parent == memo) {
    return true;
  #if ENABLE_ANCESTRY_MEMO
  } else if (gen % (unsigned)ANCESTRY_MEMO_DELTA == 0u && a.contains(memo)) {
    return true;
  #endif
  } else {
    bool result = parent->hasAncestor(memo);
    #if ENABLE_ANCESTRY_MEMO
    if (result && gen % (unsigned)ANCESTRY_MEMO_DELTA == 0u) {
      a.insert(memo);
    }
    #endif
    return result;
  }
}

libbirch::LazyAny* libbirch::LazyMemo::get(LazyAny* o, LazyMemo* from) {
  if (this == from) {
    return o;
  } else {
    o = getParent()->source(o, from);
    auto result = m.get(o);
    if (result) {
      return result;
    } else if (!o->isFrozen()) {
      return o;
    } else {
      return copy(o);
    }
  }
}

libbirch::LazyAny* libbirch::LazyMemo::pull(LazyAny* o, LazyMemo* from) {
  if (this == from) {
    return o;
  } else {
    o = getParent()->source(o, from);
    auto result = m.get(o);
    if (result) {
      return result;
    } else {
      return o;
    }
  }
}

libbirch::LazyAny* libbirch::LazyMemo::finish(LazyAny* o, LazyMemo* from) {
  if (this == from) {
    return o;
  } else {
    o = getParent()->source(o, from);
    auto result = m.get(o);
    if (result) {
      return result;
    } else {
      /* analogous to EagerMemo::copy(), see notes there */
      auto alloc = static_cast<LazyAny*>(allocate(o->getSize()));
      assert(alloc);
      auto uninit = m.uninitialized_put(o, alloc);
      assert(uninit == alloc);  // should be no thread contention here
      SwapClone swapClone(true);
      SwapFinish swapFinish(true);
      SwapContext swapContext(this);
      result = o->clone_(uninit);
      assert(result == uninit); // clone should be in the allocation
      o->incMemo(); // uninitialized_put(), so responsible for ref counts
      result->incShared();
      return result;
    }
  }
}

libbirch::LazyAny* libbirch::LazyMemo::source(LazyAny* o, LazyMemo* from) {
  if (this == from) {
    return o;
  } else {
    LazyAny* result = nullptr;
    #if ENABLE_CLONE_MEMO
    if (gen % (unsigned)CLONE_MEMO_DELTA == 0u) {
      result = m.get(o);
      if (!result) {
        result = getParent()->source(o, from);
        if (result != o) {  // if result == o then we already tried above
          result = m.get(result, result);
        }
        result = m.put(o, result);
      }
    } else {
      result = getParent()->source(o, from);
      result = m.get(result, result);
    }
    #else
    result = getParent()->source(o, from);
    result = m.get(result, result);
    #endif
    return result;
  }
}

libbirch::LazyAny* libbirch::LazyMemo::copy(LazyAny* o) {
  /* for a lazy deep clone there is no risk of infinite recursion, but
   * there may be thread contention if two threads access the same object
   * and both trigger a lazy clone simultaneously; in this case multiple
   * new objects may be made but only one thread can be successful in
   * inserting an object into the map; a shared pointer is used to
   * destroy any additional objects */
  SwapClone swapClone(true);
  SwapContext swapContext(this);
  assert(o->isFrozen());
  SharedPtr<LazyAny> cloned = o->clone_();
  // ^ use shared to clean up if beaten by another thread
  return m.put(o, cloned.get());
}

libbirch::LazyMemo* libbirch::LazyMemo::getForward() {
  if (isFrozen()) {
    auto forward = this->forward.load(std::memory_order_relaxed);
    if (!forward) {
      auto forward1 = this->create_(this);
      forward1->incShared();
      if (this->forward.compare_exchange_strong(forward, forward1,
          std::memory_order_relaxed)) {
        return forward1;
      } else {
        /* beaten by another thread */
        forward1->decShared();
      }
    }
    return forward->getForward();
  } else {
    return this;
  }
}

libbirch::LazyMemo* libbirch::LazyMemo::pullForward() {
  if (isFrozen()) {
    LazyMemo* forward = this->forward.load(std::memory_order_relaxed);
    if (forward) {
      return forward->pullForward();
    }
  }
  return this;
}

#endif
