/**
 * @file
 */
#include "bi/statement/Function.hpp"

#include "bi/visitor/all.hpp"

bi::Function::Function(Name* name, Expression* parens, Type* returnType,
    Statement* braces, Location* loc) :
    Statement(loc),
    Named(name),
    Parenthesised(parens),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

bi::Function::~Function() {
  //
}

bi::Statement* bi::Function::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Function::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Function::accept(Visitor* visitor) const {
  visitor->visit(this);
}
