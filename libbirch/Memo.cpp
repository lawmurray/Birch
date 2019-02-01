/**
 * @file
 */
#include "libbirch/Memo.hpp"

#include "libbirch/SwapClone.hpp"
#include "libbirch/SwapContext.hpp"

bi::Memo::Memo() :
    parent(nullptr),
    gen(0u) {
  //
}

bi::Memo::Memo(Memo* parent) :
    parent(parent),
    gen(parent->gen + 1u) {
  assert(parent);
}


bool bi::Memo::hasAncestor(Memo* memo) {
  if (gen <= memo->gen) {
    return false;
  } else if (parent == memo) {
    return true;
  } else if (gen % 2u == 0u && a.contains(memo)) {
    return true;
  } else {
    bool result = parent->hasAncestor(memo);
    if (result && gen % 2u == 0u) {
      a.insert(memo);
    }
    return result;
  }
}

std::pair<bi::Any*,bi::Memo*> bi::Memo::get(Any* o, Memo* from) {
  if (this == from) {
    return std::make_pair(o->getForward(), this);
  } else {
    o = getParent()->source(o, from);
    auto result = m.get(o);
    if (result) {
      return std::make_pair(result->getForward(), this);
    } else {
      return std::make_pair(copy(o), this);
    }
  }
}

std::pair<bi::Any*,bi::Memo*> bi::Memo::getNoForward(Any* o, Memo* from) {
  if (this == from) {
    return std::make_pair(o, this);
  } else {
    o = getParent()->source(o, from);
    auto result = m.get(o);
    if (result) {
      return std::make_pair(result, this);
    } else {
      return std::make_pair(copy(o), this);
    }
  }
}

std::pair<bi::Any*,bi::Memo*> bi::Memo::pull(Any* o, Memo* from) {
  if (this == from) {
    return std::make_pair(o->pullForward(), this);
  } else {
    o = getParent()->source(o, from);
    auto result = m.get(o);
    if (result) {
      return std::make_pair(result->pullForward(), this);
    } else {
      return std::make_pair(o, getParent());
    }
  }
}

bi::Any* bi::Memo::source(Any* o, Memo* from) {
  if (this == from) {
    return o;
  } else {
    Any* result = nullptr;
    if (gen % 2u == 0u) {
      result = m.get(o);
    }
    if (!result) {
      result = getParent()->source(o, from);
      if (result != o) {
        result = m.get(result, result);
      }
      if (gen % 2u == 0u) {
        result = m.put(o, result);
      }
    }
    return result;
  }
}

bi::Any* bi::Memo::copy(Any* o) {
  Any* result = nullptr;
  #if USE_LAZY_DEEP_CLONE
  /* for a lazy deep clone there is no risk of infinite recursion, but
   * there may be thread contention if two threads access the same object
   * and both trigger a lazy clone simultaneously; in this case multiple
   * new objects may be made but only one thread can be successful in
   * inserting an object into the map; a shared pointer is used to
   * destroy any additional objects */
  SwapClone swapClone(true);
  SwapContext swapContext(this);
  assert(o->isFrozen());
  SharedPtr<Any> cloned = o->clone();
  // ^ use shared to clean up if beaten by another thread
  result = m.put(o, cloned.get());
  #else
  /* for an eager deep clone we must be cautious to avoid infinite
   * recursion; memory for the new object is allocated first and put
   * in the map in case of deeper pointers back to the same object; then
   * the new object is constructed; there is no risk of another thread
   * accessing the uninitialized memory as the deep clone is not
   * accessible to other threads until completion; the new object will
   * at least have completed the Counted() constructor to initialize
   * reference counts before any recursive clones occur */
  Any* alloc = static_cast<Any*>(allocate(o->getSize()));
  assert(alloc);
  Any* uninit = m.uninitialized_put(o, alloc);
  assert(uninit == alloc);  // should be no thread contention here
  SwapClone swapClone(true);
  SwapContext swapContext(this);
  result = o->clone(uninit);
  assert(result == uninit);// clone should be in the allocation
  o->incMemo();// uninitialized_put(), so responsible for ref counts
  result->incShared();
  #endif
  return result;
}
