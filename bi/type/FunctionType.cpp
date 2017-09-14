/**
 * @file
 */
#include "bi/type/FunctionType.hpp"

#include "bi/visitor/all.hpp"

bi::FunctionType::FunctionType(Type* params, Type* returnType, Location* loc,
    const bool assignable) :
    Type(loc, assignable),
    ReturnTyped(returnType),
    params(params) {
  //
}

bi::FunctionType::~FunctionType() {
  //
}

bool bi::FunctionType::isFunction() const {
  return true;
}

bi::Type* bi::FunctionType::resolve(Type* args) {
  if (args->definitely(*params)) {
    return returnType;
  } else {
    return nullptr;
  }
}

bi::Type* bi::FunctionType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::FunctionType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::FunctionType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::FunctionType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::FunctionType::definitely(const AliasType& o) const {
  return definitely(*o.target->base);
}

bool bi::FunctionType::definitely(const FunctionType& o) const {
  return params->definitely(*o.params)
      && returnType->definitely(*o.returnType);
}

bool bi::FunctionType::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bool bi::FunctionType::definitely(const ParenthesesType& o) const {
  return definitely(*o.single);
}

bool bi::FunctionType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::FunctionType::possibly(const AliasType& o) const {
  return possibly(*o.target->base);
}

bool bi::FunctionType::possibly(const FunctionType& o) const {
  return params->possibly(*o.params) && returnType->possibly(*o.returnType);
}

bool bi::FunctionType::possibly(const OptionalType& o) const {
  return possibly(*o.single);
}

bool bi::FunctionType::possibly(const ParenthesesType& o) const {
  return possibly(*o.single);
}
