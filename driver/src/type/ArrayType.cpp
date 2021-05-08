/**
 * @file
 */
#include "src/type/ArrayType.hpp"

#include "src/visitor/all.hpp"

birch::ArrayType::ArrayType(Type* single, const int ndims,
    Location* loc) :
    Type(loc),
    Single<Type>(single),
    ndims(ndims) {
  //
}

void birch::ArrayType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

int birch::ArrayType::depth() const {
  return ndims;
}

bool birch::ArrayType::isArray() const {
  return true;
}

birch::Type* birch::ArrayType::element() {
  return single->element();
}

const birch::Type* birch::ArrayType::element() const {
  return single->element();
}
