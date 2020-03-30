/**
 * @file
 */
#include "libbirch/Label.hpp"

libbirch::Label::Label() : Any(0) {
  //
}

libbirch::Label::Label(const Label& o) {
  auto& o1 = const_cast<Label&>(o);
  o1.lock.write();
  o1.memo.rehash();
  o1.lock.downgrade();
  memo.copy(o1.memo);
  o1.lock.unread();
}

libbirch::Any* libbirch::Label::mapGet(Any* o) {
  Any* prev = nullptr;
  Any* next = o;
  bool frozen = o->isFrozen();
  while (frozen && next) {
    prev = next;
    next = memo.get(prev);
    if (next) {
      frozen = next->isFrozen();
    }
  }
  if (!next) {
	  next = prev;
	}
  if (frozen) {
    Any* cloned;
    if (next->isUnique()) {
      /* final-reference optimization */
      cloned = next->recycle(this);
    } else {
      /* copy it */
      cloned = next->copy(this);
      if (true || !next->isFrozenUnique()) {
        memo.put(next, cloned);
        thaw();
      }
    }
    next = cloned;
  }
  return next;
}

libbirch::Any* libbirch::Label::mapPull(Any* o) {
  Any* prev = nullptr;
  Any* next = o;
  bool frozen = o->isFrozen();
  while (frozen && next) {
    prev = next;
    next = memo.get(prev);
    if (next) {
      frozen = next->isFrozen();
    }
  }
  if (!next) {
	  next = prev;
	}
  return next;
}
