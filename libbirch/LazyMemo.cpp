/**
 * @file
 */
#if USE_LAZY_DEEP_CLONE
#include "libbirch/LazyMemo.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

bi::LazyMemo::LazyMemo() :
    parent(nullptr),
    gen(0u) {
  //
}

bi::LazyMemo::LazyMemo(LazyMemo* parent) :
    parent(parent),
    gen(parent->gen + 1u) {
  assert(parent);
}


bool bi::LazyMemo::hasAncestor(LazyMemo* memo) {
  if (gen <= memo->gen) {
    return false;
  } else if (parent == memo) {
    return true;
  } else if (gen % 2u == 0u && a.contains(memo)) {
    return true;
  } else {
    bool result = parent->hasAncestor(memo);
    if (result && gen % 2u == 0u) {
      a.insert(memo);
    }
    return result;
  }
}

bi::LazyAny* bi::LazyMemo::get(LazyAny* o, LazyMemo* from) {
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

bi::LazyAny* bi::LazyMemo::pull(LazyAny* o, LazyMemo* from) {
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

bi::LazyAny* bi::LazyMemo::source(LazyAny* o, LazyMemo* from) {
  if (this == from) {
    return o;
  } else {
    LazyAny* result = nullptr;
    if (gen % 2u == 0u) {
      result = m.get(o);
    }
    if (!result) {
      result = getParent()->source(o, from);
      if (result != o) {
        result = m.get(result, result);
      }
      if (gen % 2u == 0u) {
        result = m.put(o, result);
      }
    }
    return result;
  }
}

bi::LazyAny* bi::LazyMemo::copy(LazyAny* o) {
  /* for a lazy deep clone there is no risk of infinite recursion, but
   * there may be thread contention if two threads access the same object
   * and both trigger a lazy clone simultaneously; in this case multiple
   * new objects may be made but only one thread can be successful in
   * inserting an object into the map; a shared pointer is used to
   * destroy any additional objects */
  SwapClone swapClone(true);
  SwapContext swapContext(this);
  assert(o->isFrozen());
  SharedPtr<LazyAny> cloned = o->clone();
  // ^ use shared to clean up if beaten by another thread
  return m.put(o, cloned.get());
}

#endif
