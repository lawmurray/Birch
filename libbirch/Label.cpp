/**
 * @file
 */
#include "libbirch/Label.hpp"

libbirch::Label::Label() : Any(0) {
  //
}

libbirch::Label::Label(const Label& o) :
    Any(o) {
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
      /* final-reference optimization: the pointer being updated is the final
       * remaining pointer to the object, rather than copying the object and
       * then destroying it, recycle the object to be the copy */
      cloned = next->recycle(this);
    } else {
      /* copy the object */
      cloned = next->copy(this);

      /* single-reference optimization: at the time of freezing, if there was
       * only one reference to the object, it need not be memoized, as there
       * are no other pointers to update to the copy */
      if (!next->isFrozenUnique()) {
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
