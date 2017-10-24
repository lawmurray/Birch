/**
 * @file
 */
#include "bi/type/Type.hpp"

#include "bi/common/Iterator.hpp"
#include "bi/exception/all.hpp"

#include <cassert>

bi::Type::Type(Location* loc, const bool assignable) :
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

bool bi::Type::isBasic() const {
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

bool bi::Type::isList() const {
  return false;
}

bool bi::Type::isFunction() const {
  return false;
}

bool bi::Type::isFiber() const {
  return false;
}

bool bi::Type::isOptional() const {
  return false;
}

bool bi::Type::isBinary() const {
  return false;
}

bool bi::Type::isOverloaded() const {
  return false;
}

bi::Type* bi::Type::getLeft() const {
  assert(false);
  return nullptr;
}

bi::Type* bi::Type::getRight() const {
  assert(false);
  return nullptr;
}

bi::Class* bi::Type::getClass() const {
  assert(false);
  return nullptr;
}

bi::Basic* bi::Type::getBasic() const {
  assert(false);
  return nullptr;
}

bi::Type* bi::Type::unwrap() const {
  assert(false);
  return nullptr;
}

bi::FunctionType* bi::Type::resolve(Argumented* args) {
  throw CallException(args);
}

void bi::Type::resolveConstructor(Type* args) {
  if (!args->isEmpty()) {
    throw ConstructorException(args);
  }
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

bool bi::Type::definitely(const AliasType& o) const {
  return false;
}

bool bi::Type::definitely(const ArrayType& o) const {
  return false;
}

bool bi::Type::definitely(const BasicType& o) const {
  return false;
}

bool bi::Type::definitely(const BinaryType& o) const {
  return false;
}

bool bi::Type::definitely(const ClassType& o) const {
  return false;
}

bool bi::Type::definitely(const FiberType& o) const {
  return false;
}

bool bi::Type::definitely(const EmptyType& o) const {
  return false;
}

bool bi::Type::definitely(const FunctionType& o) const {
  return false;
}

bool bi::Type::definitely(const TypeIdentifier& o) const {
  return false;
}

bool bi::Type::definitely(const TypeList& o) const {
  return false;
}

bool bi::Type::definitely(const NilType& o) const {
  return false;
}

bool bi::Type::definitely(const OptionalType& o) const {
  return false;
}

bool bi::Type::definitely(const OverloadedType& o) const {
  return false;
}

bool bi::Type::definitely(const TupleType& o) const {
  return false;
}

bool bi::Type::possibly(const Type& o) const {
  return o.dispatchPossibly(*this);
}

bool bi::Type::possibly(const AliasType& o) const {
  return false;
}

bool bi::Type::possibly(const ArrayType& o) const {
  return false;
}

bool bi::Type::possibly(const BasicType& o) const {
  return false;
}

bool bi::Type::possibly(const BinaryType& o) const {
  return false;
}

bool bi::Type::possibly(const ClassType& o) const {
  return false;
}

bool bi::Type::possibly(const FiberType& o) const {
  return false;
}

bool bi::Type::possibly(const EmptyType& o) const {
  return false;
}

bool bi::Type::possibly(const FunctionType& o) const {
  return false;
}

bool bi::Type::possibly(const TypeIdentifier& o) const {
  return false;
}

bool bi::Type::possibly(const TypeList& o) const {
  return false;
}

bool bi::Type::possibly(const NilType& o) const {
  return false;
}

bool bi::Type::possibly(const OptionalType& o) const {
  return false;
}

bool bi::Type::possibly(const OverloadedType& o) const {
  return false;
}

bool bi::Type::possibly(const TupleType& o) const {
  return false;
}

bool bi::Type::equals(const Type& o) const {
  return definitely(o) && o.definitely(*this);
}
