/**
 * @file
 */
#include "bi/expression/Super.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Super::Super(shared_ptr<Location> loc) :
    Expression(loc) {
  //
}

bi::Super::~Super() {
  //
}

bi::Expression* bi::Super::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Super::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Super::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Super::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Super::definitely(const Super& o) const {
  return type->definitely(*o.type);
}

bool bi::Super::definitely(const VarParameter& o) const {
  return type->definitely(*o.type);
}

bool bi::Super::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Super::possibly(const Super& o) const {
  return type->possibly(*o.type);
}

bool bi::Super::possibly(const VarParameter& o) const {
  return type->possibly(*o.type);
}
