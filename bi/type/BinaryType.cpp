/**
 * @file
 */
#include "bi/type/BinaryType.hpp"

#include "bi/visitor/all.hpp"

bi::BinaryType::BinaryType(Type* left, Type* right, Location* loc) :
    Type(loc),
    Couple<Type>(left, right) {
  //
}

bi::BinaryType::~BinaryType() {
  //
}

bool bi::BinaryType::isBinary() const {
  return true;
}

bi::Type* bi::BinaryType::getLeft() const {
  return left;
}

bi::Type* bi::BinaryType::getRight() const {
  return right;
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
