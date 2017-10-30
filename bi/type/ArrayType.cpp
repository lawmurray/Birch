/**
 * @file
 */
#include "bi/type/ArrayType.hpp"

#include "bi/visitor/all.hpp"

bi::ArrayType::ArrayType(Type* single, Expression* brackets,
    Location* loc, const bool assignable) :
    Type(loc, assignable),
    Single<Type>(single),
    Bracketed(brackets),
    ndims(brackets->count()) {
  //
}

bi::ArrayType::ArrayType(Type* single, const int ndims,
    Location* loc, const bool assignable) :
    Type(loc, assignable),
    Single<Type>(single),
    Bracketed(new EmptyExpression(loc)),
    ndims(ndims) {
  //
}

bi::ArrayType::~ArrayType() {
  //
}

int bi::ArrayType::dims() const {
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

void bi::ArrayType::resolveConstructor(Argumented* o) {
  single->resolveConstructor(o);
}

bool bi::ArrayType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::ArrayType::definitely(const AliasType& o) const {
  assert(o.target);
  return definitely(*o.target->base);
}

bool bi::ArrayType::definitely(const GenericType& o) const {
  assert(o.target);
  return definitely(*o.target->type);
}

bool bi::ArrayType::definitely(const ArrayType& o) const {
  return single->definitely(*o.single) && ndims == o.ndims;
}

bool bi::ArrayType::definitely(const OptionalType& o) const {
  return definitely(*o.single);
}

bool bi::ArrayType::definitely(const EmptyType& o) const {
  return true;
}
