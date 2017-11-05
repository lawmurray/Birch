/**
 * @file
 */
#include "bi/type/Type.hpp"

#include "bi/type/TypeIterator.hpp"
#include "bi/type/TypeConstIterator.hpp"
#include "bi/exception/all.hpp"

#include <cassert>

bi::Type::Type(Location* loc) :
    Located(loc) {
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

int bi::Type::count() const {
  int result = 0;
  for (auto iter = begin(); iter != end(); ++iter) {
    ++result;
  }
  return result;
}

int bi::Type::dims() const {
  return 0;
}

bi::Type* bi::Type::unwrap() {
  return this;
}

const bi::Type* bi::Type::unwrap() const {
  return this;
}

bi::Type* bi::Type::canonical() {
  return this;
}

const bi::Type* bi::Type::canonical() const {
  return this;
}

bi::FunctionType* bi::Type::resolve(Argumented* o) {
  throw CallException(o);
}

void bi::Type::resolveConstructor(Argumented* o) {
  if (!o->args->isEmpty()) {
    throw ConstructorException(o);
  }
}

bi::TypeIterator bi::Type::begin() {
  if (isEmpty()) {
    return end();
  } else {
    return TypeIterator(this);
  }
}

bi::TypeConstIterator bi::Type::begin() const {
  if (isEmpty()) {
    return end();
  } else {
    return TypeConstIterator(this);
  }
}

bi::TypeIterator bi::Type::end() {
  return TypeIterator(nullptr);
}

bi::TypeConstIterator bi::Type::end() const {
  return TypeConstIterator(nullptr);
}

bool bi::Type::equals(const Type& o) const {
  return definitely(o) && o.definitely(*this);
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

bool bi::Type::definitely(const AnyType& o) const {
  return false;
}

bool bi::Type::definitely(const EmptyType& o) const {
  return false;
}

bool bi::Type::definitely(const FiberType& o) const {
  return false;
}

bool bi::Type::definitely(const FunctionType& o) const {
  return false;
}

bool bi::Type::definitely(const GenericType& o) const {
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

bool bi::Type::definitely(const SequenceType& o) const {
  return false;
}

bool bi::Type::definitely(const TupleType& o) const {
  return false;
}

bool bi::Type::definitely(const TypeIdentifier& o) const {
  return false;
}

bool bi::Type::definitely(const TypeList& o) const {
  return false;
}

bi::Type* bi::Type::common(const Type& o) const {
  return o.dispatchCommon(*this);
}

bi::Type* bi::Type::common(const AliasType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const ArrayType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const BasicType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const BinaryType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const ClassType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const AnyType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const EmptyType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const FiberType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const FunctionType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const GenericType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const NilType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const OptionalType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const OverloadedType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const SequenceType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const TupleType& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const TypeIdentifier& o) const {
  return nullptr;
}

bi::Type* bi::Type::common(const TypeList& o) const {
  return nullptr;
}
