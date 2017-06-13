/**
 * @file
 */
#include "bi/expression/MemberVariable.hpp"

#include "bi/visitor/all.hpp"

bi::MemberVariable::MemberVariable(shared_ptr<Name> name, Type* type,
    shared_ptr<Location> loc) :
    Expression(type, loc),
    Named(name) {
  //
}

bi::MemberVariable::~MemberVariable() {
  //
}

bi::Expression* bi::MemberVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::MemberVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::MemberVariable::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::MemberVariable::definitely(const MemberVariable& o) const {
  return type->definitely(*o.type);
}

bool bi::MemberVariable::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::MemberVariable::possibly(const MemberVariable& o) const {
  return type->possibly(*o.type);
}
