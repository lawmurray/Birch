/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#include "libbirch/LazyContext.hpp"

#include "libbirch/SwapContext.hpp"

libbirch::LazyAny* libbirch::LazyContext::get(LazyAny* o) {
  assert(o->isFrozen());
  LazyAny* prev = nullptr;
  LazyAny* next = o;
  bool frozen = true;
  l.set();
  do {
    prev = next;
    next = m.get(prev);
    if (next) {
      frozen = next->isFrozen();
    }
  } while (frozen && next);
  if (!next) {
	  next = prev;
	}
  if (frozen) {
    if (next->numShared() == 1u && next->numWeak() == 1u && next->numMemo() == 1u) {
      /* this is the last pointer to the object, just thaw it and reuse */
      SwapContext swapContext(this);
      next->thaw(this);
    } else {
      /* copy it */
      next = copy(next);
    }
  }
  l.unset();
  return next;
}

libbirch::LazyAny* libbirch::LazyContext::pull(LazyAny* o) {
  assert(o->isFrozen());
  LazyAny* prev = nullptr;
  LazyAny* next = o;
  bool frozen = true;
  l.set();
  do {
    prev = next;
    next = m.get(prev);
    if (next) {
      frozen = next->isFrozen();
    }
  } while (frozen && next);
  if (!next) {
	  next = prev;
	}
  l.unset();
  return next;
}

libbirch::LazyAny* libbirch::LazyContext::copy(LazyAny* o) {
  assert(o->isFrozen());
  SwapContext swapContext(this);
  auto cloned = o->clone_();
  if (!o->isSingle()) {
    thaw();  // new entry, so no longer considered frozen
    m.put(o, cloned);
  }
  return cloned;
}

void libbirch::LazyContext::freeze() {
  if (!frozen) {
    frozen = true;
    l.set();
    m.freeze();
    l.unset();
  }
}

void libbirch::LazyContext::thaw() {
  frozen = false;
}

#endif
