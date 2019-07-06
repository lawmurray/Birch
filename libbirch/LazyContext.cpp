/**
 * @file
 */
#if ENABLE_LAZY_DEEP_CLONE
#include "libbirch/LazyContext.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

libbirch::LazyAny* libbirch::LazyContext::get(LazyAny* o) {
  assert(o->isFrozen());
  LazyAny* prev = nullptr;
  LazyAny* next = o;
  l.read();
  do {
    prev = next;
    next = m.get(prev, prev);
  } while (next != prev && next->isFrozen());
  l.unread();
  if (next->isFrozen()) {
    next = copy(next);
  }
  return next;
}

libbirch::LazyAny* libbirch::LazyContext::pull(LazyAny* o) {
  assert(o->isFrozen());
  LazyAny* prev = nullptr;
  LazyAny* next = o;
  l.read();
  do {
    prev = next;
    next = m.get(prev, prev);
  } while (next != prev && next->isFrozen());
  l.unread();
  return next;
}

libbirch::LazyAny* libbirch::LazyContext::copy(LazyAny* o) {
  assert(o->isFrozen());
  SwapClone swapClone(true);
  SwapContext swapContext(this);
  auto cloned = o->clone_();
  if (!o->isSingular() || o->isMemo()) {
    cloned->memoize();
    l.write();
    frozen = false;  // no longer frozen, as will have new entry
    m.put(o, cloned);
    l.unwrite();
  }
  return cloned;
}

void libbirch::LazyContext::freeze() {
  l.read();
  if (!frozen) {
    frozen = true;
    m.freeze();
  }
  l.unread();
}

#endif
