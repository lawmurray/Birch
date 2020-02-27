/**
 * @file
 */
#include "libbirch/Label.hpp"

libbirch::Label::Label(Label* parent) :
    frozen(parent->frozen) {
  assert(parent);
  parent->lock.write();
  parent->memo.rehash();
  parent->lock.downgrade();
  memo.copy(parent->memo);
  parent->lock.unread();
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
    if (next->numShared() == 1u && next->numWeak() == 1u && next->numMemoKey() == 1u) {
      /* this is the last pointer to the object, just thaw it and reuse */
      ///@todo
      //next->thaw(this);
    } else {
      /* copy it */
      next = copy(next);
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

libbirch::Any* libbirch::Label::copy(Any* o) {
  assert(o->isFrozen());
  ///@todo
//  auto cloned = o->clone_(this);
//  if (!o->isSingle()) {
//    thaw();  // new entry, so no longer considered frozen
//    memo.put(o, cloned);
//  }
//  return cloned;
}

void libbirch::Label::freeze() {
  if (!frozen) {
    frozen = true;
    lock.read();
    memo.freeze();
    lock.unread();
  }
}

void libbirch::Label::thaw() {
  frozen = false;
}
