/**
 * @file
 */
#include "libbirch/Label.hpp"

libbirch::Any* libbirch::Label::get(Any* o) {
  assert(o->isFrozen());
  Any* prev = nullptr;
  Any* next = o;
  bool frozen = true;
  m.l.write();
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
      next->thaw(this);
    } else {
      /* copy it */
      next = copy(next);
    }
  }
  m.l.unwrite();
  return next;
}

libbirch::Any* libbirch::Label::pull(Any* o) {
  assert(o->isFrozen());
  Any* prev = nullptr;
  Any* next = o;
  bool frozen = true;
  m.l.read();
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
  m.l.unread();
  return next;
}

libbirch::Any* libbirch::Label::copy(Any* o) {
  assert(o->isFrozen());
  auto cloned = o->clone_(this);
  if (!o->isSingle()) {
    thaw();  // new entry, so no longer considered frozen
    m.put(o, cloned);
  }
  return cloned;
}

void libbirch::Label::freeze() {
  if (!frozen) {
    frozen = true;
    m.l.read();
    m.freeze();
    m.l.unread();
  }
}

void libbirch::Label::thaw() {
  frozen = false;
}
