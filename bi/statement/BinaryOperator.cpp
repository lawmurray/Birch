/**
 * @file
 */
#include "bi/statement/BinaryOperator.hpp"

#include "bi/visitor/all.hpp"

bi::BinaryOperator::BinaryOperator(shared_ptr<Name> name, Expression* parens,
    Type* returnType, Statement* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Parenthesised(parens),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

bi::BinaryOperator::~BinaryOperator() {
  //
}

bi::Statement* bi::BinaryOperator::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::BinaryOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BinaryOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
