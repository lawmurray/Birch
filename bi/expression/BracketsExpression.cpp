/**
 * @file
 */
#include "bi/expression/BracketsExpression.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::BracketsExpression::BracketsExpression(Expression* expr,
    Expression* brackets, shared_ptr<Location> loc) :
    Expression(loc), Bracketed(brackets), expr(expr) {
  /* pre-conditions */
  assert(expr);

  //
}

bi::BracketsExpression::~BracketsExpression() {
  //
}

bi::Expression* bi::BracketsExpression::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::BracketsExpression::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::BracketsExpression::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

#include "bi/io/bi_ostream.hpp"

bool bi::BracketsExpression::operator<=(Expression& o) {
  try {
    BracketsExpression& o1 = dynamic_cast<BracketsExpression&>(o);
    return *expr <= *o1.expr && *brackets <= *o1.brackets;
  } catch (std::bad_cast e) {
    //
  }
  try {
    VarParameter& o1 = dynamic_cast<VarParameter&>(o);
    return *type <= *o1.type && o1.capture(this);
  } catch (std::bad_cast e) {
    //
  }
  try {
    VarReference& o1 = dynamic_cast<VarReference&>(o);
    return *type <= *o1.type && o1.check(this);
  } catch (std::bad_cast e) {
    //
  }
  try {
    ParenthesesExpression& o1 = dynamic_cast<ParenthesesExpression&>(o);
    return *this <= *o1.expr;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::BracketsExpression::operator==(const Expression& o) const {
  try {
    const BracketsExpression& o1 =
        dynamic_cast<const BracketsExpression&>(o);
    return *expr == *o1.expr && *brackets == *o1.brackets;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
