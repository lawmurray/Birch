/**
 * @file
 */
#include "bi/expression/Traversal.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Traversal::Traversal(Expression* left, Expression* right,
    shared_ptr<Location> loc) :
    Expression(loc), ExpressionBinary(left, right) {
  //
}

bi::Traversal::~Traversal() {
  //
}

bi::Expression* bi::Traversal::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Traversal::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Traversal::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::Traversal::operator<=(Expression& o) {
  try {
    Traversal& o1 = dynamic_cast<Traversal&>(o);
    return *left <= *o1.left && *right <= *o1.right;
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

bool bi::Traversal::operator==(const Expression& o) const {
  try {
    const Traversal& o1 = dynamic_cast<const Traversal&>(o);
    return *left == *o1.left && *right == *o1.right;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
