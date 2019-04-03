/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#include "libbirch/LazyMemo.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

libbirch::LazyMemo::LazyMemo() :
    parent(nullptr),
    gen(1u) {
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

libbirch::LazyAny* libbirch::LazyMemo::get(LazyAny* o, LazyMemo* from) {
  if (this == from) {
    return o;
  } else {
    if (hasParent()) {
      o = getParent()->source(o, from);
    }
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
    if (hasParent()) {
      o = getParent()->source(o, from);
    }
    auto result = m.get(o);
    if (result) {
      return result;
    } else {
      return o;
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
        if (hasParent()) {
          result = getParent()->source(o, from);
        } else {
          /* this should only happen when doing cross copies */
          result = o;
        }
        if (result != o) {  // if result == o then we already tried above
          result = m.get(result, result);
        }
        result = m.put(o, result);
        ///@todo optimize put to start at index of previous search
      }
    } else {
      if (hasParent()) {
        result = getParent()->source(o, from);
      } else {
        result = o;
      }
      result = m.get(result, result);
    }
    #else
    if (hasParent()) {
      result = getParent()->source(o, from);
    } else {
      result = o;
    }
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
