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

bool bi::BracketsType::dispatchDefinitely(Type& o) {
  return o.definitely(*this);
}

bool bi::BracketsType::definitely(BracketsType& o) {
  return single->definitely(*o.single) && brackets->definitely(*o.brackets)
      && (!o.assignable || assignable);
}

bool bi::BracketsType::definitely(LambdaType& o) {
  return definitely(*o.result) && (!o.assignable || assignable);
}

bool bi::BracketsType::dispatchPossibly(Type& o) {
  return o.possibly(*this);
}

bool bi::BracketsType::possibly(BracketsType& o) {
  return single->possibly(*o.single) && brackets->possibly(*o.brackets)
      && (!o.assignable || assignable);
}

bool bi::BracketsType::possibly(LambdaType& o) {
  return possibly(*o.result) && (!o.assignable || assignable);
}
