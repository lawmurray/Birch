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

bi::Type* bi::FunctionType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::FunctionType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::FunctionType::common(const MemberType& o) const {
  return common(*o.right);
}

bi::Type* bi::FunctionType::common(const FunctionType& o) const {
  auto params1 = params->common(*o.params);
  auto returnType1 = returnType->common(*o.returnType);
  if (params1 && returnType1) {
    return new FunctionType(params1, returnType1);
  } else {
    return nullptr;
  }
}

bi::Type* bi::FunctionType::common(const OptionalType& o) const {
  auto single1 = common(*o.single);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}
