/**
 * @file
 */
#include "src/statement/SliceOperator.hpp"

#include "src/visitor/all.hpp"

birch::SliceOperator::SliceOperator(const Annotation annotation,
    Expression* params, Type* returnType, Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Parameterised(params),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

birch::Statement* birch::SliceOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::SliceOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
