/**
 * @file
 */
#include "bi/statement/MemberFunction.hpp"

#include "bi/visitor/all.hpp"

bi::MemberFunction::MemberFunction(shared_ptr<Name> name, Expression* parens,
    Type* returnType, Statement* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Parenthesised(parens),
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
