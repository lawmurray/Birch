/**
 * @file
 */
#include "bi/type/OptionalType.hpp"

#include "bi/visitor/all.hpp"

bi::OptionalType::OptionalType(Type* single, Location* loc,
    const bool assignable) :
    Type(loc, assignable),
    Single<Type>(single) {
  //
}

bi::OptionalType::~OptionalType() {
  //
}

bi::Type* bi::OptionalType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::OptionalType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::OptionalType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::OptionalType::isOptional() const {
  return true;
}

bool bi::OptionalType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::OptionalType::definitely(const AliasType& o) const {
  return definitely(*o.target->base);
}

bool bi::OptionalType::definitely(const OptionalType& o) const {
  return single->definitely(*o.single);
}

bool bi::OptionalType::definitely(const ParenthesesType& o) const {
  return definitely(*o.single);
}

bool bi::OptionalType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::OptionalType::possibly(const AliasType& o) const {
  return possibly(*o.target->base);
}

bool bi::OptionalType::possibly(const OptionalType& o) const {
  return single->possibly(*o.single);
}

bool bi::OptionalType::possibly(const ParenthesesType& o) const {
  return possibly(*o.single);
}
