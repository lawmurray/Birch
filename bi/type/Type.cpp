/**
 * @file
 */
#include "bi/type/Type.hpp"

#include "bi/common/Iterator.hpp"

#include <cassert>

bi::Type::Type(shared_ptr<Location> loc, const bool assignable) :
    Located(loc),
    assignable(assignable) {
  //
}

bi::Type::~Type() {
  //
}

bool bi::Type::isEmpty() const {
  return false;
}

bool bi::Type::isBuiltin() const {
  return false;
}

bool bi::Type::isStruct() const {
  return false;
}

bool bi::Type::isClass() const {
  return false;
}

bool bi::Type::isAlias() const {
  return false;
}

bool bi::Type::isArray() const {
  return false;
}

bool bi::Type::isFunction() const {
  return false;
}

bool bi::Type::isCoroutine() const {
  return false;
}

bi::Type* bi::Type::strip() {
  return this;
}

int bi::Type::count() const {
  return 0;
}

bi::Iterator<bi::Type> bi::Type::begin() const {
  if (isEmpty()) {
    return end();
  } else {
    return bi::Iterator<Type>(this);
  }
}

bi::Iterator<bi::Type> bi::Type::end() const {
  return bi::Iterator<Type>(nullptr);
}

bool bi::Type::definitely(const Type& o) const {
  return o.dispatchDefinitely(*this);
}

bool bi::Type::definitely(const BracketsType& o) const {
  return false;
}

bool bi::Type::definitely(const CoroutineType& o) const {
  return false;
}

bool bi::Type::definitely(const EmptyType& o) const {
  return false;
}

bool bi::Type::definitely(const FunctionType& o) const {
  return false;
}

bool bi::Type::definitely(const List<Type>& o) const {
  return false;
}

bool bi::Type::definitely(const Class& o) const {
  return false;
}

bool bi::Type::definitely(const IdentifierType<Class>& o) const {
  return false;
}

bool bi::Type::definitely(const IdentifierType<AliasType>& o) const {
  return false;
}

bool bi::Type::definitely(const IdentifierType<BasicType>& o) const {
  return false;
}

bool bi::Type::definitely(const ParenthesesType& o) const {
  return false;
}

bool bi::Type::possibly(const Type& o) const {
  return o.dispatchPossibly(*this);
}

bool bi::Type::possibly(const BracketsType& o) const {
  return false;
}

bool bi::Type::possibly(const CoroutineType& o) const {
  return false;
}

bool bi::Type::possibly(const EmptyType& o) const {
  return false;
}

bool bi::Type::possibly(const FunctionType& o) const {
  return false;
}

bool bi::Type::possibly(const List<Type>& o) const {
  return false;
}

bool bi::Type::possibly(const Class& o) const {
  return false;
}

bool bi::Type::possibly(const IdentifierType<Class>& o) const {
  return false;
}

bool bi::Type::possibly(const IdentifierType<AliasType>& o) const {
  return false;
}

bool bi::Type::possibly(const IdentifierType<BasicType>& o) const {
  return false;
}

bool bi::Type::possibly(const ParenthesesType& o) const {
  return false;
}

bool bi::Type::equals(const Type& o) const {
  return definitely(o) && o.definitely(*this);
}
