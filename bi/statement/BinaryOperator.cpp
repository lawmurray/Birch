/**
 * @file
 */
#include "bi/statement/BinaryOperator.hpp"

#include "bi/visitor/all.hpp"

bi::BinaryOperator::BinaryOperator(const Annotation annotation,
    Expression* left, Name* name, Expression* right, Type* returnType,
    Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    Couple<Expression>(left, right),
    ReturnTyped(returnType),
    Scoped(LOCAL_SCOPE),
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
