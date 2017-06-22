/**
 * @file
 */
#include "bi/expression/MemberParameter.hpp"

#include "bi/visitor/all.hpp"

bi::MemberParameter::MemberParameter(shared_ptr<Name> name, Type* type,
    Expression* value, shared_ptr<Location> loc) :
    Expression(type, loc),
    Named(name),
    Valued(value) {
  //
}

bi::MemberParameter::~MemberParameter() {
  //
}

bi::Expression* bi::MemberParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::MemberParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::MemberParameter::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::MemberParameter::definitely(const MemberParameter& o) const {
  return type->definitely(*o.type);
}

bool bi::MemberParameter::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::MemberParameter::possibly(const MemberParameter& o) const {
  return type->possibly(*o.type);
}
