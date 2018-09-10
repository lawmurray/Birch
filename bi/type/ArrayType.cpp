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

void bi::ArrayType::resolveConstructor(Argumented* o) {
  single->resolveConstructor(o);
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

bool bi::ArrayType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}

bool bi::ArrayType::definitely(const MemberType& o) const {
  return definitely(*o.right);
}

bool bi::ArrayType::definitely(const ArrayType& o) const {
  return single->definitely(*o.single) && depth() == o.depth();
}

bool bi::ArrayType::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bi::Type* bi::ArrayType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}

bi::Type* bi::ArrayType::common(const GenericType& o) const {
  assert(o.target);
  return common(*o.target->type);
}

bi::Type* bi::ArrayType::common(const MemberType& o) const {
  return common(*o.right);
}

bi::Type* bi::ArrayType::common(const ArrayType& o) const {
  auto single1 = single->common(*o.single);
  if (single1 && depth() == o.depth()) {
    return new ArrayType(single1, depth());
  } else {
    return nullptr;
  }
}

bi::Type* bi::ArrayType::common(const OptionalType& o) const {
  auto single1 = common(*o.single);
  if (single1) {
    return new OptionalType(single1);
  } else {
    return nullptr;
  }
}
