/**
 * @file
 */
#include "src/type/FunctionType.hpp"

#include "src/visitor/all.hpp"

birch::FunctionType::FunctionType(Type* params, Type* returnType, Location* loc) :
    Type(loc),
    ReturnTyped(returnType),
    params(params) {
  //
}

birch::FunctionType::~FunctionType() {
  //
}

birch::Type* birch::FunctionType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Type* birch::FunctionType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::FunctionType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool birch::FunctionType::isFunction() const {
  return true;
}

bool birch::FunctionType::isValue() const {
  return false;
}
