/**
 * @file
 */
#include "bi/statement/ConversionOperator.hpp"

#include "bi/visitor/all.hpp"

bi::ConversionOperator::ConversionOperator(Type* returnType, Statement* braces,
    Location* loc) :
    Statement(loc),
    ReturnTyped(returnType),
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

bi::ConversionOperator::~ConversionOperator() {
  //
}

bi::Statement* bi::ConversionOperator::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::ConversionOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ConversionOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
