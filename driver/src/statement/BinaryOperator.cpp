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
    Named(name),
    TypeParameterised(typeParams),
    Couple<Expression>(left, right),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

void birch::BinaryOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
