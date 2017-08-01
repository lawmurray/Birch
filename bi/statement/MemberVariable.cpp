/**
 * @file
 */
#include "bi/statement/MemberVariable.hpp"

#include "bi/visitor/all.hpp"

bi::MemberVariable::MemberVariable(shared_ptr<Name> name, Type* type,
    Expression* parens, Expression* value, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Typed(type),
    Parenthesised(parens),
    Valued(value) {
  //
}

bi::MemberVariable::~MemberVariable() {
  //
}

bi::Statement* bi::MemberVariable::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::MemberVariable::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberVariable::accept(Visitor* visitor) const {
  visitor->visit(this);
}
