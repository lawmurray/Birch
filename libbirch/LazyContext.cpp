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
  l.write();
  do {
    prev = next;
    next = m.get(prev, prev);
  } while (next != prev && next->isFrozen());
  if (next->isFrozen()) {
    next = copy(next);
  }
  l.unwrite();
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

libbirch::LazyAny* libbirch::LazyContext::finish(LazyAny* o) {
  assert(o->isFrozen());
  LazyAny* prev = nullptr;
  LazyAny* next = o;
  //l.keep();
  while (next != prev && this != next->getContext()) {
    prev = next;
    next = m.get(prev, prev);
  }
  if (this != next->getContext()) {
    next = copy(next);
  }
  //l.unkeep();
  return next;
}

libbirch::LazyAny* libbirch::LazyContext::copy(LazyAny* o) {
  /* for a lazy deep clone there is no risk of infinite recursion, but
   * there may be thread contention if two threads access the same object
   * and both trigger a lazy clone simultaneously; in this case multiple
   * new objects may be made but only one thread can be successful in
   * inserting an object into the map; a shared pointer is used to
   * destroy any additional objects */
  assert(o->isFrozen());
  SwapClone swapClone(true);
  SwapContext swapContext(this);
  auto cloned = o->clone_();
  if (!o->isSingular() || o->isMemo()) {
    frozen.store(true);
    cloned->memoize();
    m.put(o, cloned);
  }
  return cloned;
}

void libbirch::LazyContext::freeze() {
  if (!frozen.exchange(true)) {
    l.read();
    m.freeze();
    l.unread();
  }
}

#endif
