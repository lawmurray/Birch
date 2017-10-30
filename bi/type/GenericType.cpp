/**
 * @file
 */
#include "bi/type/GenericType.hpp"

#include "bi/visitor/all.hpp"

bi::GenericType::GenericType(Name* name, Location* loc, const bool assignable,
    Generic* target) :
    Type(loc, assignable),
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

bi::Basic* bi::GenericType::getBasic() const {
  assert(target);
  return target->type->getBasic();
}

bi::Class* bi::GenericType::getClass() const {
  assert(target);
  return target->type->getClass();
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

bool bi::GenericType::definitely(const AliasType& o) const {
  assert(target);
  return target->type->definitely(o);
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

bool bi::GenericType::definitely(const TupleType& o) const {
  assert(target);
  return target->type->definitely(o);
}

bool bi::GenericType::definitely(const TypeList& o) const {
  assert(target);
  return target->type->definitely(o);
}
