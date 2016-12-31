/**
 * @file
 */
#include "bi/expression/ParenthesesExpression.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ParenthesesExpression::ParenthesesExpression(Expression* expr, shared_ptr<Location> loc) :
    Expression(loc), expr(expr) {
  /* pre-condition */
  assert(expr);
}

bi::ParenthesesExpression::~ParenthesesExpression() {
  //
}

bi::Expression* bi::ParenthesesExpression::strip() {
  return expr->strip();
}

bi::Expression* bi::ParenthesesExpression::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::ParenthesesExpression::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ParenthesesExpression::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ParenthesesExpression::operator<=(Expression& o) {
  try {
    ParenthesesExpression& o1 = dynamic_cast<ParenthesesExpression&>(o);
    return *expr <= *o1.expr;
  } catch (std::bad_cast e) {
    //
  }
  try {
    VarParameter& o1 = dynamic_cast<VarParameter&>(o);
    return *type <= *o1.type && o1.capture(this);
  } catch (std::bad_cast e) {
    //
  }

  /* parentheses may be used unnecessarily in situations where precedence is
   * clear anyway; accommodate these by making their use optional in
   * matches */
  return *expr <= o;
}

bool bi::ParenthesesExpression::operator==(const Expression& o) const {
  return *expr == o;
}
