/**
 * @file
 */
#include "bi/statement/Class.hpp"

#include "bi/visitor/all.hpp"

bi::Class::Class(shared_ptr<Name> name, Type* base, Statement* braces,
    shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Based(base),
    Braced(braces) {
  //
}

bi::Class::~Class() {
  //
}

bi::Statement* bi::Class::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Class::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Class::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Class::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::Class::definitely(const Class& o) const {
  return true;
}

bool bi::Class::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::Class::possibly(const Class& o) const {
  return true;
}
