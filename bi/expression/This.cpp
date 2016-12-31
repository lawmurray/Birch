/**
 * @file
 */
#include "bi/expression/This.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::This::This(shared_ptr<Location> loc) :
    Expression(loc) {
  //
}

bi::This::~This() {
  //
}

bi::Expression* bi::This::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::This::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::This::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::This::operator<=(Expression& o) {
  try {
    This& o1 = dynamic_cast<This&>(o);
    return *type <= *o1.type;
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
    ParenthesesExpression& o1 = dynamic_cast<ParenthesesExpression&>(o);
    return *this <= *o1.expr;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::This::operator==(const Expression& o) const {
  try {
    const This& o1 = dynamic_cast<const This&>(o);
    return *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
