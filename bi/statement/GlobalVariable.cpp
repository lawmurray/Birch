/**
 * @file
 */
#include "bi/statement/GlobalVariable.hpp"

#include "bi/visitor/all.hpp"

bi::GlobalVariable::GlobalVariable(shared_ptr<Name> name, Type* type,
    shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Typed(type) {
  //
}

bi::GlobalVariable::~GlobalVariable() {
  //
}

bi::Statement* bi::GlobalVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::GlobalVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::GlobalVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::GlobalVariable::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::GlobalVariable::definitely(const GlobalVariable& o) const {
  return type->definitely(*o.type);
}

bool bi::GlobalVariable::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::GlobalVariable::possibly(const GlobalVariable& o) const {
  return type->possibly(*o.type);
}
