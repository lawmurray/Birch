/**
 * @file
 */
#include "src/statement/ConversionOperator.hpp"

#include "src/visitor/all.hpp"

birch::ConversionOperator::ConversionOperator(const Annotation annotation,
    Type* returnType, Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

void birch::ConversionOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
