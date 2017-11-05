/**
 * @file
 */
#include "bi/type/SequenceType.hpp"

#include "bi/visitor/all.hpp"

bi::SequenceType::SequenceType(Type* single, Location* loc) :
    Type(loc),
    Single<Type>(single) {
  //
}

bi::SequenceType::~SequenceType() {
  //
}

bi::Type* bi::SequenceType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::SequenceType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::SequenceType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::SequenceType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::SequenceType::definitely(const AliasType& o) const {
  assert(o.target);
  return definitely(*o.target->base);
}

bool bi::SequenceType::definitely(const ArrayType& o) const {
  return single->definitely(*o.single);
}

bool bi::SequenceType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}

bool bi::SequenceType::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bool bi::SequenceType::definitely(const SequenceType& o) const {
  return single->definitely(*o.single);
}

bool bi::SequenceType::definitely(const AnyType& o) const {
  return true;
}
