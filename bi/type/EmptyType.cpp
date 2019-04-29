/**
 * @file
 */
#include "bi/type/EmptyType.hpp"

#include "bi/visitor/all.hpp"

bi::EmptyType::EmptyType(Location* loc) :
    Type(loc) {
  //
}

bi::EmptyType::~EmptyType() {
  //
}

bi::Type* bi::EmptyType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::EmptyType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::EmptyType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::EmptyType::isValue() const {
  return true;
}

bool bi::EmptyType::isEmpty() const {
  return true;
}

bool bi::EmptyType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::EmptyType::isConvertible(const EmptyType& o) const {
  return true;
}

bool bi::EmptyType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::EmptyType::isAssignable(const EmptyType& o) const {
  return true;
}

bi::Type* bi::EmptyType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::EmptyType::common(const EmptyType& o) const {
  return new EmptyType();
}
