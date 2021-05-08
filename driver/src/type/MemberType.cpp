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

birch::Type* birch::MemberType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::MemberType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

int birch::MemberType::depth() const {
  return right->depth();
}
