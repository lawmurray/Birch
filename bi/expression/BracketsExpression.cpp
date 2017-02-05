/**
 * @file
 */
#include "bi/expression/BracketsExpression.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::BracketsExpression::BracketsExpression(Expression* single,
    Expression* brackets, shared_ptr<Location> loc) :
    Expression(loc), ExpressionUnary(single), Bracketed(brackets) {
  //
}

bi::BracketsExpression::~BracketsExpression() {
  //
}

bi::Expression* bi::BracketsExpression::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::BracketsExpression::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BracketsExpression::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::BracketsExpression::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::BracketsExpression::definitely(BracketsExpression& o) {
  return single->definitely(*o.single) && brackets->definitely(*o.brackets);
}

bool bi::BracketsExpression::definitely(VarParameter& o) {
  return type->definitely(*o.type) && o.capture(this);
}

bool bi::BracketsExpression::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::BracketsExpression::possibly(BracketsExpression& o) {
  return single->possibly(*o.single) && brackets->possibly(*o.brackets);
}

bool bi::BracketsExpression::possibly(VarParameter& o) {
  return type->possibly(*o.type) && o.capture(this);
}
