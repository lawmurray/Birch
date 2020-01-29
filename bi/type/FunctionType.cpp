/**
 * @file
 */
#include "bi/type/FunctionType.hpp"

#include "bi/visitor/all.hpp"

bi::FunctionType::FunctionType(Type* params, Type* returnType, Location* loc) :
    Type(loc),
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

bi::Type* bi::FunctionType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::FunctionType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::FunctionType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::FunctionType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::FunctionType::isConvertible(const GenericType& o) const {
  assert(o.target);
  return isConvertible(*o.target->type);
}

bool bi::FunctionType::isConvertible(const MemberType& o) const {
  return isConvertible(*o.right);
}

bool bi::FunctionType::isConvertible(const FunctionType& o) const {
  return params->isConvertible(*o.params)
      && returnType->isConvertible(*o.returnType);
}

bool bi::FunctionType::isConvertible(const OptionalType& o) const {
  return isConvertible(*o.single);
}

bool bi::FunctionType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::FunctionType::isAssignable(const ClassType& o) const {
  return o.getClass()->hasAssignment(this);
}

bool bi::FunctionType::isAssignable(const GenericType& o) const {
  assert(o.target);
  return isAssignable(*o.target->type);
}

bool bi::FunctionType::isAssignable(const MemberType& o) const {
  return isAssignable(*o.right);
}

bool bi::FunctionType::isAssignable(const FunctionType& o) const {
  return params->isAssignable(*o.params)
      && returnType->isAssignable(*o.returnType);
}

bool bi::FunctionType::isAssignable(const OptionalType& o) const {
  return isAssignable(*o.single);
}
