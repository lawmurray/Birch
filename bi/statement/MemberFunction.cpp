/**
 * @file
 */
#include "bi/statement/MemberFunction.hpp"

#include "bi/visitor/all.hpp"

bi::MemberFunction::MemberFunction(Name* name, Expression* params,
    Type* returnType, Statement* braces, Location* loc) :
    Statement(loc),
    Named(name),
    Parameterised(params),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

bi::MemberFunction::~MemberFunction() {
  //
}

bi::Statement* bi::MemberFunction::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::MemberFunction::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberFunction::accept(Visitor* visitor) const {
  visitor->visit(this);
}
