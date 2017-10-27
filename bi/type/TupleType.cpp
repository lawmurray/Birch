/**
 * @file
 */
#include "bi/type/TupleType.hpp"

#include "bi/visitor/all.hpp"

bi::TupleType::TupleType(Type* single, Location* loc,
    const bool assignable) :
    Type(loc, assignable),
    Single<Type>(single) {
  //
}

bi::TupleType::~TupleType() {
  //
}

bi::Type* bi::TupleType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::TupleType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::TupleType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::TupleType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::TupleType::definitely(const AliasType& o) const {
  assert(o.target);
  return definitely(*o.target->base);
}

bool bi::TupleType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}

bool bi::TupleType::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bool bi::TupleType::definitely(const TupleType& o) const {
  return single->definitely(*o.single);
}
