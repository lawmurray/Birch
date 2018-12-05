/**
 * @file
 */
#include "libbirch/Memo.hpp"

bi::Memo::Memo(Memo* parent) :
    parent(parent) {
  //
}

bi::Memo::~Memo() {
  //
}

bool bi::Memo::hasAncestor(Memo* memo) const {
  SharedPtr<Memo> parent = this->parent;
  return parent && (parent == memo || parent->hasAncestor(memo));
}

bi::Memo* bi::Memo::forward() {
  return child ? child->forward() : this;
}

bi::Memo* bi::Memo::fork() {
  #if DEEP_CLONE_STRATEGY != DEEP_CLONE_EAGER
  if (!child) {
    /* create the forward memo */
    ///@todo make this thread safe
    child = create(this);
    if (globalMemo == this) {
      globalMemo = child;
    }
  }
  #endif

  /* create and return the clone memo */
  return create(this);
}
