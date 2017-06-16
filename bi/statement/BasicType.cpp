/**
 * @file
 */
#include "bi/statement/BasicType.hpp"

#include "bi/visitor/all.hpp"

bi::BasicType::BasicType(shared_ptr<Name> name, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name) {
  //
}

bi::BasicType::~BasicType() {
  //
}

bi::Statement* bi::BasicType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::BasicType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BasicType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::BasicType::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::BasicType::definitely(const BasicType& o) const {
  return true;
}

bool bi::BasicType::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::BasicType::possibly(const BasicType& o) const {
  return true;
}
