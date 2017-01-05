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

bi::Expression* bi::VarParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::VarParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::VarParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::VarParameter::dispatch(Expression& o) {
  return o.le(*this);
}

bool bi::VarParameter::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}
