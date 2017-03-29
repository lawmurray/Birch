/**
 * @file
 */
#include "bi/expression/EmptyExpression.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::EmptyExpression::EmptyExpression() {
  //
}

bi::EmptyExpression::~EmptyExpression() {
  //
}

bi::Expression* bi::EmptyExpression::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::EmptyExpression::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::EmptyExpression::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::EmptyExpression::isEmpty() const {
  return true;
}

bool bi::EmptyExpression::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::EmptyExpression::definitely(const EmptyExpression& o) const {
  return type->definitely(*o.type);
}

bool bi::EmptyExpression::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::EmptyExpression::possibly(const EmptyExpression& o) const {
  return type->possibly(*o.type);
}
