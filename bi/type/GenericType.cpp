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

bool bi::GenericType::isPointer() const {
  assert(target);
  return target->type->isPointer();
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

bi::FunctionType* bi::GenericType::resolve(Argumented* o) {
  assert(target);
  return target->type->resolve(o);
}

void bi::GenericType::resolveConstructor(Argumented* o) {
  assert(target);
  target->type->resolveConstructor(o);
}

bool bi::GenericType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::GenericType::definitely(const ArrayType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const BasicType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const ClassType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const AnyType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const EmptyType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const FiberType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const FunctionType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const GenericType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const OptionalType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const SequenceType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const TupleType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const TypeList& o) const {
  assert(target);
  return target->type->definitely(o);
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

bi::Type* bi::GenericType::common(const AnyType& o) const {
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

bi::Type* bi::GenericType::common(const OptionalType& o) const {
  assert(target);
  return target->type->common(o);
}

bi::Type* bi::GenericType::common(const SequenceType& o) const {
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
