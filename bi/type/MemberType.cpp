/**
 * @file
 */
#include "bi/type/MemberType.hpp"

#include "bi/visitor/all.hpp"

bi::MemberType::MemberType(Type* left, Type* right, Location* loc) :
    Type(loc),
    Couple<Type>(left, right) {
  //
}

bi::MemberType::~MemberType() {
  //
}

bi::Type* bi::MemberType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::MemberType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::MemberType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

int bi::MemberType::depth() const {
  return right->depth();
}
