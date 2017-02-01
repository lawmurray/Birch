/**
 * @file
 */
#include "bi/type/ModelParameter.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ModelParameter::ModelParameter(shared_ptr<Name> name, Expression* parens,
    shared_ptr<Name> op, Type* base, Expression* braces,
    shared_ptr<Location> loc) :
    Type(loc),
    Named(name),
    Parenthesised(parens),
    Based(op, base),
    Braced(braces) {
  this->arg = this;
}

bi::ModelParameter::~ModelParameter() {
  //
}

bi::Type* bi::ModelParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::ModelParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ModelParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ModelParameter::isBuiltin() const {
  if (*op == "=") {
    return base->isBuiltin();
  } else {
    return braces->isEmpty();
  }
}

bool bi::ModelParameter::isModel() const {
  if (*op == "=") {
    return base->isModel();
  } else {
    return !braces->isEmpty();
  }
}

bi::possibly bi::ModelParameter::dispatch(Type& o) {
  return o.le(*this);
}

bi::possibly bi::ModelParameter::le(ModelParameter& o) {
  return *parens <= *o.parens && *base <= *o.base
      && (!o.assignable || assignable) && o.capture(this);
}

bi::possibly bi::ModelParameter::le(AssignableType& o) {
  return *this <= *o.single;
}

bi::possibly bi::ModelParameter::le(ParenthesesType& o) {
  return *this <= *o.single;
}

bi::possibly bi::ModelParameter::le(EmptyType& o) {
  return possibly(!o.assignable || assignable);
}
