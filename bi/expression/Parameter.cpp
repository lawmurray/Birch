/**
 * @file
 */
#include "bi/expression/Parameter.hpp"

#include "bi/visitor/all.hpp"

bi::Parameter::Parameter(shared_ptr<Name> name, Type* type, Expression* value,
    shared_ptr<Location> loc) :
    Expression(type, loc),
    Named(name),
    Valued(value) {
  //
}

bi::Parameter::~Parameter() {
  //
}

bi::Expression* bi::Parameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Parameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Parameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Parameter::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Parameter::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::Parameter::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Parameter::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}
