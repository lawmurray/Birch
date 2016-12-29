/**
 * @file
 */
#include "bi/expression/Range.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Range::Range(Expression* left, Expression* right,
    shared_ptr<Location> loc) :
    Expression(loc), ExpressionBinary(left, right) {
  //
}

bi::Expression* bi::Range::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

void bi::Range::acceptModify(Modifier* visitor) {
  visitor->modify(this);
}

void bi::Range::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::Range::operator<=(Expression& o) {
  try {
    Range& o1 = dynamic_cast<Range&>(o);
    return *left <= *o1.left && *right <= *o1.right;
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

bool bi::Range::operator==(const Expression& o) const {
  try {
    const Range& o1 = dynamic_cast<const Range&>(o);
    return *left == *o1.left && *right == *o1.right;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
