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
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

birch::ConversionOperator::~ConversionOperator() {
  //
}

birch::Statement* birch::ConversionOperator::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::ConversionOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::ConversionOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
