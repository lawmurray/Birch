/**
 * @file
 */
#include "bi/expression/BracesExpression.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::BracesExpression::BracesExpression(Statement* stmt,
    shared_ptr<Location> loc) :
    Expression(loc), stmt(stmt) {
  //
}

bi::Expression* bi::BracesExpression::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::BracesExpression::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::BracesExpression::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::BracesExpression::operator<=(Expression& o) {
  try {
    BracesExpression& o1 = dynamic_cast<BracesExpression&>(o);
    return *stmt <= *o1.stmt && *type <= *o1.type;
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

bool bi::BracesExpression::operator==(const Expression& o) const {
  try {
    const BracesExpression& o1 = dynamic_cast<const BracesExpression&>(o);
    return *stmt == *o1.stmt && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
