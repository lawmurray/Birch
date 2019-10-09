/**
 * @file
 */
#include "bi/type/GenericType.hpp"

#include "bi/visitor/all.hpp"

bi::GenericType::GenericType(Name* name, Location* loc, Generic* target) :
    Type(loc),
    Named(name),
    Reference<Generic>(target) {
  //
}

bi::GenericType::GenericType(Generic* target) :
    Named(target->name),
    Reference<Generic>(target) {
  //
}

bi::GenericType::~GenericType() {
  //
}

bi::Type* bi::GenericType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::GenericType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::GenericType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::GenericType::isBasic() const {
  assert(target);
  return target->type->isBasic();
}

bool bi::GenericType::isClass() const {
  assert(target);
  return target->type->isClass();
}

bool bi::GenericType::isWeak() const {
  assert(target);
  return target->type->isWeak();
}

bool bi::GenericType::isArray() const {
  assert(target);
  return target->type->isArray();
}

bool bi::GenericType::isFunction() const {
  assert(target);
  return target->type->isFunction();
}

bool bi::GenericType::isFiber() const {
  assert(target);
  return target->type->isFiber();
}

bool bi::GenericType::isGeneric() const {
  return true;
}

const bi::Type* bi::GenericType::unwrap() const {
  assert(target);
  return target->type->unwrap();
}

bi::Type* bi::GenericType::unwrap() {
  assert(target);
  return target->type->unwrap();
}

int bi::GenericType::depth() const {
  assert(target);
  return target->type->depth();
}

bi::Basic* bi::GenericType::getBasic() const {
  assert(target);
  return target->type->getBasic();
}

bi::Class* bi::GenericType::getClass() const {
  assert(target);
  return target->type->getClass();
}

bi::Type* bi::GenericType::canonical() {
  assert(target);
  return target->type->canonical();
}

const bi::Type* bi::GenericType::canonical() const {
  assert(target);
  return target->type->canonical();
}

void bi::GenericType::resolveConstructor(Argumented* o) {
  assert(target);
  target->type->resolveConstructor(o);
}

bool bi::GenericType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bool bi::GenericType::isConvertible(const ArrayType& o) const {
  assert(target);
  return target->type->isConvertible(o);
}

bool bi::GenericType::isConvertible(const BasicType& o) const {
  assert(target);
  return target->type->isConvertible(o);
}

bool bi::GenericType::isConvertible(const ClassType& o) const {
  assert(target);
  return target->type->isConvertible(o);
}

bool bi::GenericType::isConvertible(const EmptyType& o) const {
  assert(target);
  return target->type->isConvertible(o);
}

bool bi::GenericType::isConvertible(const FiberType& o) const {
  assert(target);
  return target->type->isConvertible(o);
}

bool bi::GenericType::isConvertible(const FunctionType& o) const {
  assert(target);
  return target->type->isConvertible(o);
}

bool bi::GenericType::isConvertible(const GenericType& o) const {
  assert(target);
  return target->type->isConvertible(o);
}

bool bi::GenericType::isConvertible(const MemberType& o) const {
  assert(target);
  return target->type->isConvertible(o);
}

bool bi::GenericType::isConvertible(const OptionalType& o) const {
  assert(target);
  return target->type->isConvertible(o);
}

bool bi::GenericType::isConvertible(const TupleType& o) const {
  assert(target);
  return target->type->isConvertible(o);
}

bool bi::GenericType::isConvertible(const TypeList& o) const {
  assert(target);
  return target->type->isConvertible(o);
}

bool bi::GenericType::dispatchIsAssignable(const Type& o) const {
  return o.isAssignable(*this);
}

bool bi::GenericType::isAssignable(const ArrayType& o) const {
  assert(target);
  return target->type->isAssignable(o);
}

bool bi::GenericType::isAssignable(const BasicType& o) const {
  assert(target);
  return target->type->isAssignable(o);
}

bool bi::GenericType::isAssignable(const ClassType& o) const {
  assert(target);
  return target->type->isAssignable(o);
}

bool bi::GenericType::isAssignable(const EmptyType& o) const {
  assert(target);
  return target->type->isAssignable(o);
}

bool bi::GenericType::isAssignable(const FiberType& o) const {
  assert(target);
  return target->type->isAssignable(o);
}

bool bi::GenericType::isAssignable(const FunctionType& o) const {
  assert(target);
  return target->type->isAssignable(o);
}

bool bi::GenericType::isAssignable(const GenericType& o) const {
  assert(target);
  return target->type->isAssignable(o);
}

bool bi::GenericType::isAssignable(const MemberType& o) const {
  assert(target);
  return target->type->isAssignable(o);
}

bool bi::GenericType::isAssignable(const OptionalType& o) const {
  assert(target);
  return target->type->isAssignable(o);
}

bool bi::GenericType::isAssignable(const TupleType& o) const {
  assert(target);
  return target->type->isAssignable(o);
}

bool bi::GenericType::isAssignable(const TypeList& o) const {
  assert(target);
  return target->type->isAssignable(o);
}

bi::Type* bi::GenericType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::GenericType::common(const ArrayType& o) const {
  assert(target);
  return target->type->common(o);
}

bi::Type* bi::GenericType::common(const BasicType& o) const {
  assert(target);
  return target->type->common(o);
}

bi::Type* bi::GenericType::common(const ClassType& o) const {
  assert(target);
  return target->type->common(o);
}

bi::Type* bi::GenericType::common(const EmptyType& o) const {
  assert(target);
  return target->type->common(o);
}

bi::Type* bi::GenericType::common(const FiberType& o) const {
  assert(target);
  return target->type->common(o);
}

bi::Type* bi::GenericType::common(const FunctionType& o) const {
  assert(target);
  return target->type->common(o);
}

bi::Type* bi::GenericType::common(const GenericType& o) const {
  assert(target);
  return target->type->common(o);
}

bi::Type* bi::GenericType::common(const MemberType& o) const {
  assert(target);
  return target->type->common(o);
}

bi::Type* bi::GenericType::common(const OptionalType& o) const {
  assert(target);
  return target->type->common(o);
}

bi::Type* bi::GenericType::common(const TupleType& o) const {
  assert(target);
  return target->type->common(o);
}

bi::Type* bi::GenericType::common(const TypeList& o) const {
  assert(target);
  return target->type->common(o);
}
