/**
 * @file
 */
#include "bi/type/BinaryType.hpp"

#include "bi/visitor/all.hpp"

bi::BinaryType::BinaryType(Type* left, Type* right, Location* loc,
    const bool assignable) :
    Type(loc, assignable),
    Couple<Type>(left, right) {
  //
}

bi::BinaryType::~BinaryType() {
  //
}

bi::Type* bi::BinaryType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::BinaryType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BinaryType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::BinaryType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::BinaryType::definitely(const BinaryType& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::BinaryType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::BinaryType::possibly(const BinaryType& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}
