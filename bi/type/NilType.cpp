/**
 * @file
 */
#include "bi/type/NilType.hpp"

#include "bi/visitor/all.hpp"

bi::NilType::NilType(Location* loc) :
    Type(loc) {
  //
}

bi::NilType::~NilType() {
  //
}

bi::Type* bi::NilType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::NilType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::NilType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::NilType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::NilType::isConvertible(const NilType& o) const {
  return true;
}

bool bi::NilType::isConvertible(const OptionalType& o) const {
  return true;
}

bool bi::NilType::isConvertible(const WeakType& o) const {
  return true;
}

bool bi::NilType::isConvertible(const SequenceType& o) const {
  return true;
}

bool bi::NilType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::NilType::isAssignable(const NilType& o) const {
  return true;
}

bool bi::NilType::isAssignable(const OptionalType& o) const {
  return true;
}

bool bi::NilType::isAssignable(const WeakType& o) const {
  return true;
}

bool bi::NilType::isAssignable(const SequenceType& o) const {
  return true;
}

bi::Type* bi::NilType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::NilType::common(const NilType& o) const {
  return new NilType();
}

bi::Type* bi::NilType::common(const OptionalType& o) const {
  return o.common(o);
}

bi::Type* bi::NilType::common(const WeakType& o) const {
  return o.common(o);
}

bi::Type* bi::NilType::common(const SequenceType& o) const {
  return o.common(o);
}
