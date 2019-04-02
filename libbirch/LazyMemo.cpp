/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#include "libbirch/LazyMemo.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

libbirch::LazyMemo::LazyMemo() :
    parent(nullptr),
    gen(0u) {
  //
}

libbirch::LazyMemo::LazyMemo(LazyMemo* parent) :
    parent(parent),
    gen(parent->gen + 1u) {
  assert(parent);
}

libbirch::LazyMemo::~LazyMemo() {
  //
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
      if (this->isFrozen()) {
        result->freeze();
      }
      return result;
    }
  }
}

libbirch::LazyAny* libbirch::LazyMemo::cross(LazyAny* o) {
  auto result = m.get(o);
  if (!result) {
    /* analogous to EagerMemo::copy(), see notes there */
    auto alloc = static_cast<LazyAny*>(allocate(o->getSize()));
    assert(alloc);
    auto uninit = m.uninitialized_put(o, alloc);
    assert(uninit == alloc);  // should be no thread contention here
    SwapClone swapClone(true);
    SwapCross swapCross(true);
    SwapContext swapContext(this);
    result = o->clone_(uninit);
    assert(result == uninit); // clone should be in the allocation
    o->incMemo(); // uninitialized_put(), so responsible for ref counts
    result->incShared();
    if (this->isFrozen()) {
      result->freeze();
    }
  }
  return result;
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
        ///@todo optimize put to start at index of previous search
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
  auto result = m.put(o, cloned.get());
  if (this->isFrozen()) {
    result->freeze();
  }
  return result;
}

#endif
