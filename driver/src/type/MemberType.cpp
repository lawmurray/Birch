/**
 * @file
 */
#include "src/type/MemberType.hpp"

#include "src/visitor/all.hpp"

birch::MemberType::MemberType(Type* left, Type* right, Location* loc) :
    Type(loc),
    Couple<Type>(left, right) {
  //
}

birch::MemberType::~MemberType() {
  //
}

birch::Type* birch::MemberType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Type* birch::MemberType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::MemberType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

int birch::MemberType::depth() const {
  return right->depth();
}

bool birch::MemberType::isMembership() const {
  return true;
}

bool birch::MemberType::isValue() const {
  return right->isValue();
}
