/**
 * @file
 */
#include "src/statement/UnaryOperator.hpp"

#include "src/visitor/all.hpp"

birch::UnaryOperator::UnaryOperator(const Annotation annotation, Expression* typeParams,
    Name* name, Expression* single, Type* returnType, Statement* braces,
    Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Named(name),
    TypeParameterised(typeParams),
    Single<Expression>(single),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

void birch::UnaryOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
