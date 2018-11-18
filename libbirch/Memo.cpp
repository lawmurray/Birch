/**
 * @file
 */
#include "libbirch/Memo.hpp"

#include "libbirch/memory.hpp"

bi::Memo::Memo(Memo* parent) :
    parent(parent) {
  //
}

bi::Memo::~Memo() {
  //
}

bool bi::Memo::hasAncestor(Memo* memo) const {
  return parent == memo || (parent && parent->hasAncestor(memo));
}

bi::Memo* bi::Memo::getParent() {
  return parent;
}
