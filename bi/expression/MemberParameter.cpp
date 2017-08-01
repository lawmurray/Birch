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
