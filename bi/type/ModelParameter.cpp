/**
 * @file
 */
#include "bi/type/ModelParameter.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ModelParameter::ModelParameter(shared_ptr<Name> name, Expression* parens,
    shared_ptr<Name> op, Type* base, Expression* braces,
    shared_ptr<Location> loc, const bool assignable) :
    Type(loc, assignable),
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
  if (isEqual()) {
    return base->isBuiltin();
  } else {
    return braces->isEmpty();
  }
}

bool bi::ModelParameter::isModel() const {
  if (isEqual()) {
    return base->isModel();
  } else {
    return !braces->isEmpty();
  }
}

bool bi::ModelParameter::isLess() const {
  return !base->isEmpty() && *op == "<";
}

bool bi::ModelParameter::isEqual() const {
  return !base->isEmpty() && *op == "=";
}

bool bi::ModelParameter::dispatchDefinitely(Type& o) {
  return o.definitely(*this);
}

bool bi::ModelParameter::definitely(ModelParameter& o) {
  return parens->definitely(*o.parens) && base->definitely(*o.base)
      && (!o.assignable || assignable) && o.capture(this);
}

bool bi::ModelParameter::definitely(EmptyType& o) {
  return !o.assignable || assignable;
}

bool bi::ModelParameter::dispatchPossibly(Type& o) {
  return o.possibly(*this);
}

bool bi::ModelParameter::possibly(ModelParameter& o) {
  return parens->possibly(*o.parens) && base->possibly(*o.base)
      && (!o.assignable || assignable) && o.capture(this);
}

bool bi::ModelParameter::possibly(EmptyType& o) {
  return !o.assignable || assignable;
}
