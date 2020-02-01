/**
 * @file
 */
#include "bi/type/ArrayType.hpp"

#include "bi/visitor/all.hpp"

bi::ArrayType::ArrayType(Type* single, const int ndims,
    Location* loc) :
    Type(loc),
    Single<Type>(single),
    ndims(ndims) {
  //
}

bi::ArrayType::~ArrayType() {
  //
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

int bi::ArrayType::depth() const {
  return ndims;
}

bool bi::ArrayType::isArray() const {
  return true;
}

bi::Type* bi::ArrayType::element() {
  return single->element();
}

const bi::Type* bi::ArrayType::element() const {
  return single->element();
}
