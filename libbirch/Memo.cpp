/**
 * @file
 */
#include "libbirch/Memo.hpp"

#include "libbirch/memory.hpp"

bi::Memo::Memo() :
    parent(nullptr) {
  //
}

bi::Memo::Memo(int) :
    parent(nullptr) {
  incShared();
}

bi::Memo::Memo(Memo* parent) :
    parent(parent) {
  //
}

bi::Memo::~Memo() {
  //
}

bi::Memo* bi::Memo::clone() const {
  return make_object<Memo>(*this);
}

void bi::Memo::destroy() {
  size = sizeof(*this);
  this->~Memo();
}

bool bi::Memo::hasAncestor(Memo* memo) const {
  return this == memo || (parent && parent->hasAncestor(memo));
}

bi::Memo* bi::Memo::getParent() {
  return parent;
}
