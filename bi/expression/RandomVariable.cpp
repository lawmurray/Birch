/**
 * @file
 */
#include "bi/expression/RandomVariable.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::RandomVariable::RandomVariable(Expression* left, Expression* right,
    shared_ptr<Location> loc) :
    Expression(loc), ExpressionBinary(left, right) {
  //
}

bi::RandomVariable::~RandomVariable() {
  //
}

bi::Expression* bi::RandomVariable::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::RandomVariable::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::RandomVariable::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::RandomVariable::operator<=(Expression& o) {
  try {
    RandomVariable& o1 = dynamic_cast<RandomVariable&>(o);
    return *left <= *o1.left && *right <= *o1.right;
  } catch (std::bad_cast e) {
    //
  }
  try {
    VarParameter& o1 = dynamic_cast<VarParameter&>(o);
    return *left->type <= *o1.type && o1.capture(left.get());
  } catch (std::bad_cast e) {
    //
  }
  try {
    VarReference& o1 = dynamic_cast<VarReference&>(o);
    return *left->type <= *o1.type && o1.check(left.get());
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

bool bi::RandomVariable::operator==(const Expression& o) const {
  try {
    const RandomVariable& o1 = dynamic_cast<const RandomVariable&>(o);
    return *left == *o1.left && *right == *o1.right;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
