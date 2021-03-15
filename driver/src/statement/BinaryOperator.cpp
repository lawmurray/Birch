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
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

birch::BinaryOperator::~BinaryOperator() {
  //
}

bool birch::BinaryOperator::isDeclaration() const {
  return true;
}

birch::Statement* birch::BinaryOperator::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::BinaryOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::BinaryOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
