/**
 * @file
 */
#include "bi/expression/VarParameter.hpp"

#include "bi/expression/VarReference.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::VarParameter::VarParameter(shared_ptr<Name> name, Type* type,
    Expression* value, shared_ptr<Location> loc) :
    Expression(type, loc),
    Named(name),
    value(value) {
  //
}

bi::VarParameter::~VarParameter() {
  //
}

bi::Expression* bi::VarParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::VarParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::VarParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::VarParameter::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::VarParameter::definitely(const VarParameter& o) const {
  return type->definitely(*o.type);
}

bool bi::VarParameter::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::VarParameter::possibly(const VarParameter& o) const {
  return type->possibly(*o.type);
}
