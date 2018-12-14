/**
 * @file
 */
#include "libbirch/Memo.hpp"

bi::Memo::Memo() :
    isForked(false) {
  //
}

bi::Memo::Memo(Memo* parent, const bool isForwarding) :
    cloneParent(isForwarding ? nullptr : parent),
    forwardParent(isForwarding ? parent : nullptr),
    isForked(false) {
  //
}

bi::Memo::~Memo() {
  //
}

bool bi::Memo::hasAncestor(Memo* memo) const {
  SharedPtr<Memo> parent = getParent();
  return parent && (parent == memo || parent->hasAncestor(memo));
}

bi::Memo* bi::Memo::fork() {
  #if DEEP_CLONE_STRATEGY != DEEP_CLONE_EAGER
  isForked = true;
  #endif
  return create(this, false);
}

bi::Memo* bi::Memo::forwardGet() {
  #if DEEP_CLONE_STRATEGY != DEEP_CLONE_EAGER
  ///@todo make this thread safe
  if (forwardChild) {
    return forwardChild->forwardGet();
  } else if (isForked) {
    forwardChild = create(this, true);
    return forwardChild.get();
  } else {
    return this;
  }
  #else
  return this;
  #endif
}

bi::Memo* bi::Memo::forwardPull() {
  #if DEEP_CLONE_STRATEGY != DEEP_CLONE_EAGER
  return forwardChild ? forwardChild->forwardPull() : this;
  #else
  return this;
  #endif
}

bi::SharedPtr<bi::Memo> bi::Memo::getParent() const {
  return cloneParent ? cloneParent : SharedPtr<Memo>(forwardParent);
}
