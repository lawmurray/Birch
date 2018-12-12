/**
 * @file
 */
#include "libbirch/Memo.hpp"

bi::Memo::Memo(Memo* parent) :
    parent(parent),
    forked(false) {
  if (parent) {
    parent->children.push_back(this);
  }
}

bi::Memo::~Memo() {
  for (auto child: children) {
    SharedPtr<Memo> child1 = child;
    if (child1) {
      child1->collect();
    }
  }
}

bool bi::Memo::hasAncestor(Memo* memo) const {
  SharedPtr<Memo> parent = this->parent;
  return parent && (parent == memo || parent->hasAncestor(memo));
}

bi::Memo* bi::Memo::forwardGet() {
  #if DEEP_CLONE_STRATEGY != DEEP_CLONE_EAGER
  ///@todo make this thread safe
  if (child) {
    return child->forwardGet();
  } else if (forked) {
    child = create(this);
    return child.get();
  } else {
    return this;
  }
  #else
  return this;
  #endif
}

bi::Memo* bi::Memo::forwardPull() {
  #if DEEP_CLONE_STRATEGY != DEEP_CLONE_EAGER
  return child ? child->forwardPull() : this;
  #else
  return this;
  #endif
}

bi::Memo* bi::Memo::fork() {
  #if DEEP_CLONE_STRATEGY != DEEP_CLONE_EAGER
  forked = true;
  #endif
  return create(this);
}

void bi::Memo::collect() {
  clones.collect();
}
