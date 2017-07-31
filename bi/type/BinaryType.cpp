/**
 * @file
 */
#include "bi/type/BinaryType.hpp"

#include "bi/visitor/all.hpp"

bi::BinaryType::BinaryType(Type* left, Type* right, Type* returnType,
    shared_ptr<Location> loc, const bool assignable) :
    Type(loc, assignable),
    Binary<Type>(left, right),
    ReturnTyped(returnType) {
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
  return visitor->visit(this);
}

bool bi::BinaryType::isBinary() const {
  return true;
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
