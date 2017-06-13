/**
 * @file
 */
#include "bi/expression/LocalVariable.hpp"

#include "bi/visitor/all.hpp"

bi::LocalVariable::LocalVariable(shared_ptr<Name> name, Type* type,
    shared_ptr<Location> loc) :
    Expression(type, loc),
    Named(name) {
  //
}

bi::LocalVariable::~LocalVariable() {
  //
}

bi::Expression* bi::LocalVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::LocalVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::LocalVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::LocalVariable::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::LocalVariable::definitely(const LocalVariable& o) const {
  return type->definitely(*o.type);
}

bool bi::LocalVariable::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::LocalVariable::possibly(const LocalVariable& o) const {
  return type->possibly(*o.type);
}
