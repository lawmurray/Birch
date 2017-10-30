/**
 * @file
 */
#include "bi/type/NilType.hpp"

#include "bi/visitor/all.hpp"

bi::NilType::NilType(Location* loc) :
    Type(loc) {
  //
}

bi::NilType::~NilType() {
  //
}

bi::Type* bi::NilType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::NilType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::NilType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::NilType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::NilType::definitely(const NilType& o) const {
  return true;
}

bool bi::NilType::definitely(const OptionalType& o) const {
  return true;
}
