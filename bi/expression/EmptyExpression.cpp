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

bi::possibly bi::EmptyExpression::dispatch(Expression& o) {
  return o.le(*this);
}

bi::possibly bi::EmptyExpression::le(EmptyExpression& o) {
  return *type <= *o.type;
}
