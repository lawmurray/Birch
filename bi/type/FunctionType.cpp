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

bi::FunctionType* bi::FunctionType::resolve(Argumented* o) {
  if (o->args->type->definitely(*params)) {
    return this;
  } else {
    throw CallException(o, this);
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

bool bi::FunctionType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}

bool bi::FunctionType::definitely(const FunctionType& o) const {
  return params->definitely(*o.params)
      && returnType->definitely(*o.returnType);
}

bool bi::FunctionType::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bool bi::FunctionType::definitely(const AnyType& o) const {
  return true;
}

bi::Type* bi::FunctionType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::FunctionType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
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

bi::Type* bi::FunctionType::common(const AnyType& o) const {
  return new AnyType();
}
