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

bool bi::GenericType::isGeneric() const {
  return true;
}

bool bi::GenericType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::GenericType::definitely(const AliasType& o) const {
  assert(o.target);
  return definitely(*o.target->base);
}

bool bi::GenericType::definitely(const GenericType& o) const {
  return target == o.target;
}

bool bi::GenericType::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bool bi::GenericType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::GenericType::possibly(const AliasType& o) const {
  assert(o.target);
  return possibly(*o.target->base);
}

bool bi::GenericType::possibly(const GenericType& o) const {
  return false;
}

bool bi::GenericType::possibly(const OptionalType& o) const {
  return possibly(*o.single);
}
