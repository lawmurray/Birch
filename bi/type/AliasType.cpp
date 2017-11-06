/**
 * @file
 */
#include "bi/type/AliasType.hpp"

#include "bi/visitor/all.hpp"

bi::AliasType::AliasType(Name* name, Location* loc, Alias* target) :
    Type(loc),
    Named(name),
    Reference<Alias>(target) {
  //
}

bi::AliasType::AliasType(Alias* target) :
    Named(target->name),
    Reference<Alias>(target) {
  //
}

bi::AliasType::~AliasType() {
  //
}

bi::Type* bi::AliasType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::AliasType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::AliasType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::AliasType::isBasic() const {
  assert(target);
  return target->base->isBasic();
}

bool bi::AliasType::isClass() const {
  assert(target);
  return target->base->isClass();
}

bool bi::AliasType::isArray() const {
  assert(target);
  return target->base->isArray();
}

bool bi::AliasType::isFunction() const {
  assert(target);
  return target->base->isFunction();
}

bool bi::AliasType::isFiber() const {
  assert(target);
  return target->base->isFiber();
}

int bi::AliasType::depth() const {
  assert(target);
  return target->base->depth();
}

bi::Basic* bi::AliasType::getBasic() const {
  assert(target);
  return target->base->getBasic();
}

bi::Class* bi::AliasType::getClass() const {
  assert(target);
  return target->base->getClass();
}

bi::Type* bi::AliasType::canonical() {
  assert(target);
  return target->base->canonical();
}

const bi::Type* bi::AliasType::canonical() const {
  assert(target);
  return target->base->canonical();
}

bi::FunctionType* bi::AliasType::resolve(Argumented* o) {
  assert(target);
  return target->base->resolve(o);
}

void bi::AliasType::resolveConstructor(Argumented* o) {
  assert(target);
  target->base->resolveConstructor(o);
}

bool bi::AliasType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::AliasType::definitely(const AliasType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const ArrayType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const BasicType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const ClassType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const AnyType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const EmptyType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const FiberType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const FunctionType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const GenericType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const OptionalType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const SequenceType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const TupleType& o) const {
  assert(target);
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const TypeList& o) const {
  assert(target);
  return target->base->definitely(o);
}

bi::Type* bi::AliasType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::AliasType::common(const AliasType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const ArrayType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const BasicType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const ClassType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const AnyType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const EmptyType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const FiberType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const FunctionType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const GenericType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const OptionalType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const SequenceType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const TupleType& o) const {
  assert(target);
  return target->base->common(o);
}

bi::Type* bi::AliasType::common(const TypeList& o) const {
  assert(target);
  return target->base->common(o);
}
