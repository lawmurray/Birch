/**
 * @file
 */
#include "bi/type/UnaryType.hpp"

#include "bi/visitor/all.hpp"

bi::UnaryType::UnaryType(Type* single, Type* returnType,
    shared_ptr<Location> loc, const bool assignable) :
    Type(loc, assignable),
    Unary<Type>(single),
    ReturnTyped(returnType) {
  //
}

bi::UnaryType::~UnaryType() {
  //
}

bi::Type* bi::UnaryType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::UnaryType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::UnaryType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::UnaryType::isUnary() const {
  return true;
}

bool bi::UnaryType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::UnaryType::definitely(const UnaryType& o) const {
  return single->definitely(*o.single);
}

bool bi::UnaryType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::UnaryType::possibly(const UnaryType& o) const {
  return single->possibly(*o.single);
}
