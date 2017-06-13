/**
 * @file
 */
#include "bi/expression/GlobalVariable.hpp"

#include "bi/visitor/all.hpp"

bi::GlobalVariable::GlobalVariable(shared_ptr<Name> name, Type* type,
    shared_ptr<Location> loc) :
    Expression(type, loc),
    Named(name) {
  //
}

bi::GlobalVariable::~GlobalVariable() {
  //
}

bi::Expression* bi::GlobalVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::GlobalVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::GlobalVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::GlobalVariable::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::GlobalVariable::definitely(const GlobalVariable& o) const {
  return type->definitely(*o.type);
}

bool bi::GlobalVariable::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::GlobalVariable::possibly(const GlobalVariable& o) const {
  return type->possibly(*o.type);
}
