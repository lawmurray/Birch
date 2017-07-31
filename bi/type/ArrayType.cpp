/**
 * @file
 */
#include "bi/type/ArrayType.hpp"

#include "bi/visitor/all.hpp"

bi::ArrayType::ArrayType(Type* single, Expression* brackets,
    shared_ptr<Location> loc, const bool assignable) :
    Type(loc, assignable),
    Unary<Type>(single),
    Bracketed(brackets),
    ndims(brackets->tupleSize()) {
  //
}

bi::ArrayType::ArrayType(Type* single, const int ndims,
    shared_ptr<Location> loc, const bool assignable) :
    Type(loc, assignable),
    Unary<Type>(single),
    ndims(ndims) {
  //
}

bi::ArrayType::~ArrayType() {
  //
}

int bi::ArrayType::count() const {
  return ndims;
}

bool bi::ArrayType::isArray() const {
  return true;
}

bi::Type* bi::ArrayType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::ArrayType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ArrayType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::ArrayType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::ArrayType::definitely(const ArrayType& o) const {
  return single->definitely(*o.single) && ndims == o.ndims;
}

bool bi::ArrayType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::ArrayType::possibly(const ArrayType& o) const {
  return single->possibly(*o.single) && ndims == o.ndims;
}
