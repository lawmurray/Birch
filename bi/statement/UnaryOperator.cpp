/**
 * @file
 */
#include "bi/statement/UnaryOperator.hpp"

#include "bi/visitor/all.hpp"

bi::UnaryOperator::UnaryOperator(shared_ptr<Name> name, Expression* parens,
    Type* returnType, Statement* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Parenthesised(parens),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

bi::UnaryOperator::~UnaryOperator() {
  //
}

bi::Statement* bi::UnaryOperator::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::UnaryOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::UnaryOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
