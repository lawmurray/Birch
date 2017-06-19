/**
 * @file
 */
#include "bi/type/AliasType.hpp"

#include "bi/visitor/all.hpp"

bi::AliasType::AliasType(shared_ptr<Name> name, shared_ptr<Location> loc,
    const bool assignable, const Alias* target) :
    Type(loc, assignable),
    Named(name),
    Reference<Alias>(target) {
  //
}

bi::AliasType::AliasType(const Alias* target) :
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

bool bi::AliasType::isCoroutine() const {
  return target->base->isCoroutine();
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

bool bi::AliasType::definitely(const CoroutineType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const EmptyType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const FunctionType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const List<Type>& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const ParenthesesType& o) const {
  return target->base->definitely(o);
}

bool bi::AliasType::definitely(const Alias& o) const {
  return true;
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

bool bi::AliasType::possibly(const CoroutineType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const EmptyType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const FunctionType& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const List<Type>& o) const {
  return target->base->possibly(o);
}

bool bi::AliasType::possibly(const ParenthesesType& o) const {
  return target->base->possibly(o);
}
