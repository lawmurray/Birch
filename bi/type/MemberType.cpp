/**
 * @file
 */
#include "bi/type/MemberType.hpp"

#include "bi/visitor/all.hpp"

bi::MemberType::MemberType(Type* left, Type* right, Location* loc) :
    Type(loc),
    Couple<Type>(left, right) {
  //
}

bi::MemberType::~MemberType() {
  //
}

bi::Type* bi::MemberType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::MemberType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::MemberType::isBasic() const {
  return right->isBasic();
}

bool bi::MemberType::isClass() const {
  return right->isClass();
}

bool bi::MemberType::isWeak() const {
  return right->isWeak();
}

bool bi::MemberType::isArray() const {
  return right->isArray();
}

bool bi::MemberType::isFunction() const {
  return right->isFunction();
}

bool bi::MemberType::isFiber() const {
  return right->isFiber();
}

bool bi::MemberType::isMember() const {
  return true;
}

const bi::Type* bi::MemberType::unwrap() const {
  return right->unwrap();
}

bi::Type* bi::MemberType::unwrap() {
  return right->unwrap();
}

int bi::MemberType::depth() const {
  return right->depth();
}

bi::Basic* bi::MemberType::getBasic() const {
  return right->getBasic();
}

bi::Class* bi::MemberType::getClass() const {
  return right->getClass();
}

bi::Type* bi::MemberType::canonical() {
  return right->canonical();
}

const bi::Type* bi::MemberType::canonical() const {
  return right->canonical();
}

void bi::MemberType::resolveConstructor(Argumented* o) {
  right->resolveConstructor(o);
}

bool bi::MemberType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::MemberType::isConvertible(const ArrayType& o) const {
  return right->isConvertible(o);
}

bool bi::MemberType::isConvertible(const BasicType& o) const {
  return right->isConvertible(o);
}

bool bi::MemberType::isConvertible(const ClassType& o) const {
  return right->isConvertible(o);
}

bool bi::MemberType::isConvertible(const EmptyType& o) const {
  return right->isConvertible(o);
}

bool bi::MemberType::isConvertible(const FiberType& o) const {
  return right->isConvertible(o);
}

bool bi::MemberType::isConvertible(const FunctionType& o) const {
  return right->isConvertible(o);
}

bool bi::MemberType::isConvertible(const GenericType& o) const {
  return right->isConvertible(o);
}

bool bi::MemberType::isConvertible(const MemberType& o) const {
  return right->isConvertible(o);
}

bool bi::MemberType::isConvertible(const OptionalType& o) const {
  return right->isConvertible(o);
}

bool bi::MemberType::isConvertible(const TupleType& o) const {
  return right->isConvertible(o);
}

bool bi::MemberType::isConvertible(const TypeList& o) const {
  return right->isConvertible(o);
}

bool bi::MemberType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::MemberType::isAssignable(const ArrayType& o) const {
  return right->isAssignable(o);
}

bool bi::MemberType::isAssignable(const BasicType& o) const {
  return right->isAssignable(o);
}

bool bi::MemberType::isAssignable(const ClassType& o) const {
  return right->isAssignable(o);
}

bool bi::MemberType::isAssignable(const EmptyType& o) const {
  return right->isAssignable(o);
}

bool bi::MemberType::isAssignable(const FiberType& o) const {
  return right->isAssignable(o);
}

bool bi::MemberType::isAssignable(const FunctionType& o) const {
  return right->isAssignable(o);
}

bool bi::MemberType::isAssignable(const GenericType& o) const {
  return right->isAssignable(o);
}

bool bi::MemberType::isAssignable(const MemberType& o) const {
  return right->isAssignable(o);
}

bool bi::MemberType::isAssignable(const OptionalType& o) const {
  return right->isAssignable(o);
}

bool bi::MemberType::isAssignable(const TupleType& o) const {
  return right->isAssignable(o);
}

bool bi::MemberType::isAssignable(const TypeList& o) const {
  return right->isAssignable(o);
}
