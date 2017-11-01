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

bool bi::FunctionType::definitely(const AliasType& o) const {
  assert(o.target);
  return definitely(*o.target->base);
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
