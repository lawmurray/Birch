/**
 * @file
 */
#include "bi/statement/MemberVariable.hpp"

#include "bi/visitor/all.hpp"

bi::MemberVariable::MemberVariable(Name* name, Type* type,
    Expression* brackets, Expression* args, Expression* value, Location* loc) :
    Statement(loc),
    Named(name),
    Typed(type),
    Bracketed(brackets),
    Argumented(args),
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
