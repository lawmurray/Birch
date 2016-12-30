/**
 * @file
 */
#include "bi/expression/EmptyExpression.hpp"

#include "bi/type/EmptyType.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::EmptyExpression::EmptyExpression() {
  //
}

bi::EmptyExpression::~EmptyExpression() {
  //
}

bi::Expression* bi::EmptyExpression::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::EmptyExpression::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::EmptyExpression::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bi::EmptyExpression::operator bool() const {
  return false;
}

bool bi::EmptyExpression::operator<=(Expression& o) {
  try {
    EmptyExpression& o1 = dynamic_cast<EmptyExpression&>(o);
    return *type <= *o1.type;
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

bool bi::EmptyExpression::operator==(const Expression& o) const {
  try {
    const EmptyExpression& o1 = dynamic_cast<const EmptyExpression&>(o);
    return true;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
