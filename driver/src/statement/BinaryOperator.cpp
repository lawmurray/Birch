/**
 * @file
 */
#include "src/statement/BinaryOperator.hpp"

#include "src/visitor/all.hpp"

birch::BinaryOperator::BinaryOperator(const Annotation annotation,
    Expression* typeParams, Expression* left, Name* name, Expression* right,
    Type* returnType, Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    TypeParameterised(typeParams),
    Named(name),
    Couple<Expression>(left, right),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

birch::Statement* birch::BinaryOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::BinaryOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
