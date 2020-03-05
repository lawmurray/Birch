/**
 * @file
 */
#include "libbirch/Label.hpp"

libbirch::Label::Label() {
  //
}

libbirch::Label::Label(const Label& o) {
  auto o1 = const_cast<Label&>(o);
  o1.lock.write();
  o1.memo.rehash();
  o1.lock.downgrade();
  memo.copy(o1.memo);
  o1.lock.unread();
}

libbirch::Any* libbirch::Label::get(Any* o) {
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
    if (next->isUnique()) {
      /* this is the last pointer to the object, recycle it */
      ///@todo
      //next->recycle(this);
    } else {
      /* copy it */
      auto old = next;
      ///@todo
      //next = old->clone(this);
      if (!old->isSingle()) {
        memo.put(old, next);
      }
    }
  }
  return next;
}

libbirch::Any* libbirch::Label::pull(Any* o) {
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
