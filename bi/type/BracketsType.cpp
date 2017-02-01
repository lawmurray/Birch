/**
 * @file
 */
#include "bi/type/BracketsType.hpp"

#include "bi/visitor/all.hpp"

bi::BracketsType::BracketsType(Type* single, Expression* brackets,
    shared_ptr<Location> loc) :
    Type(loc),
    TypeUnary(single),
    Bracketed(brackets),
    ndims(brackets->tupleSize()) {
  //
}

bi::BracketsType::BracketsType(Type* single, const int ndims) :
    TypeUnary(single),
    ndims(ndims) {
  //
}

bi::BracketsType::~BracketsType() {
  //
}

int bi::BracketsType::count() const {
  return ndims;
}

bi::Type* bi::BracketsType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::BracketsType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BracketsType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bi::possibly bi::BracketsType::dispatch(Type& o) {
  return o.le(*this);
}

bi::possibly bi::BracketsType::le(BracketsType& o) {
  return *single <= *o.single && *brackets <= *o.brackets
      && (!o.assignable || assignable);
}

bi::possibly bi::BracketsType::le(AssignableType& o) {
  return *this <= *o.single;
}

bi::possibly bi::BracketsType::le(ParenthesesType& o) {
  return *this <= *o.single;
}
