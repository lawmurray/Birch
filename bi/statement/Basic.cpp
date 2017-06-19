/**
 * @file
 */
#include "bi/statement/Basic.hpp"

#include "bi/visitor/all.hpp"

bi::Basic::Basic(shared_ptr<Name> name, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name) {
  //
}

bi::Basic::~Basic() {
  //
}

bi::Statement* bi::Basic::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Basic::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Basic::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Basic::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::Basic::definitely(const Basic& o) const {
  return true;
}

bool bi::Basic::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::Basic::possibly(const Basic& o) const {
  return true;
}
