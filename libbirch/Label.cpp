/**
 * @file
 */
#include "libbirch/Label.hpp"

libbirch::Label::Label(Label* parent) {
  if (parent) {
    parent->lock.write();
    parent->memo.rehash();
    parent->lock.downgrade();
    memo.copy(parent->memo);
    parent->lock.unread();
  }
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
