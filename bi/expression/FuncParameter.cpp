/**
 * @file
 */
#include "bi/expression/FuncParameter.hpp"

#include "bi/expression/FuncReference.hpp"
#include "bi/primitive/encode.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::FuncParameter::FuncParameter(shared_ptr<Name> name, Expression* parens,
    Expression* result, Expression* braces, const FunctionForm form, shared_ptr<Location> loc) :
    Expression(loc),
    Named(name),
    Braced(braces),
    Parenthesised(parens),
    Formed(form),
    result(result) {
  this->arg = this;
  if (parens->isRich()) {
    this->unique = new Name(uniqueName(this));
  } else {
    this->unique = new Name(internalName(this));
  }
}

bi::FuncParameter::~FuncParameter() {
  //
}

bi::Expression* bi::FuncParameter::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::FuncParameter::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::FuncParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::FuncParameter::operator<=(Expression& o) {
  try {
    FuncParameter& o1 = dynamic_cast<FuncParameter&>(o);
    return *parens <= *o1.parens && *type <= *o1.type && o1.capture(this);
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

bool bi::FuncParameter::operator==(const Expression& o) const {
  try {
    const FuncParameter& o1 = dynamic_cast<const FuncParameter&>(o);
    return *parens == *o1.parens && *type == *o1.type;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
