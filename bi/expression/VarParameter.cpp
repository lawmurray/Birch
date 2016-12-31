/**
 * @file
 */
#include "bi/expression/VarParameter.hpp"

#include "bi/expression/VarReference.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::VarParameter::VarParameter(shared_ptr<Name> name, Type* type,
    Expression* parens, Expression* value, shared_ptr<Location> loc) :
    Expression(type, loc),
    Named(name),
    Parenthesised(parens),
    value(value) {
  this->arg = this;
}

bi::VarParameter::~VarParameter() {
  //
}

bi::Expression* bi::VarParameter::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::VarParameter::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::VarParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::VarParameter::operator<=(Expression& o) {
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

bool bi::VarParameter::operator==(const Expression& o) const {
  try {
    const VarParameter& o1 = dynamic_cast<const VarParameter&>(o);
    return *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
