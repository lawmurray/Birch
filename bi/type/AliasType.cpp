/**
 * @file
 */
#include "bi/type/AliasType.hpp"

#include "bi/visitor/all.hpp"

bi::AliasType::AliasType(Name* name, Location* loc,
    const bool assignable, Alias* target) :
    Type(loc, assignable),
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
  return target->base->isBasic();
}

bool bi::AliasType::isClass() const {
  return target->base->isClass();
}

bool bi::AliasType::isAlias() const {
  return true;
}

bool bi::AliasType::isArray() const {
  return target->base->isArray();
}

bool bi::AliasType::isFunction() const {
  return target->base->isFunction();
}

bool bi::AliasType::isFiber() const {
  return target->base->isFiber();
}

bool bi::AliasType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::AliasType::definitely(const AliasType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const ArrayType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const BasicType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const ClassType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const FiberType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const EmptyType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const FunctionType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const ListType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const OptionalType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const ParenthesesType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::AliasType::possibly(const AliasType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const ArrayType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const BasicType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const ClassType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const FiberType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const EmptyType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const FunctionType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const ListType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const OptionalType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const ParenthesesType& o) const {
  return target->base->possibly(o);
}
