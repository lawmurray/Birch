/**
 * @file
 */
#include "bi/type/EmptyType.hpp"

#include "bi/visitor/all.hpp"

bi::EmptyType::EmptyType(shared_ptr<Location> loc) :
    Type(loc) {
  //
}

bi::EmptyType::~EmptyType() {
  //
}

bi::Type* bi::EmptyType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::EmptyType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::EmptyType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::EmptyType::isEmpty() const {
  return true;
}

bi::possibly bi::EmptyType::dispatch(Type& o) {
  return o.le(*this);
}

bi::possibly bi::EmptyType::le(EmptyType& o) {
  return possibly(!o.assignable || assignable);
}
