/**
 * @file
 */
#include "src/statement/ConversionOperator.hpp"

#include "src/visitor/all.hpp"

birch::ConversionOperator::ConversionOperator(Type* returnType, Statement* braces,
    Location* loc) :
    Statement(loc),
    ReturnTyped(returnType),
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

birch::ConversionOperator::~ConversionOperator() {
  //
}

bool birch::ConversionOperator::isDeclaration() const {
  return true;
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
