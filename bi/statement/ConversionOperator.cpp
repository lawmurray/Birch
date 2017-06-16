/**
 * @file
 */
#include "bi/statement/ConversionOperator.hpp"

#include "bi/visitor/all.hpp"

bi::ConversionOperator::ConversionOperator(Type* returnType, Statement* braces,
    shared_ptr<Location> loc) :
    Statement(loc),
    ReturnTyped(returnType),
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

bool bi::ConversionOperator::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::ConversionOperator::definitely(const ConversionOperator& o) const {
  return returnType->definitely(*o.returnType);
}

bool bi::ConversionOperator::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::ConversionOperator::possibly(const ConversionOperator& o) const {
  return returnType->possibly(*o.returnType);
}
