/**
 * @file
 */
#include "bi/type/UnknownType.hpp"

#include "bi/visitor/all.hpp"

bi::UnknownType::UnknownType(const bool weak, Name* name,
    Type* typeArgs, Location* loc) :
    Type(loc),
    Named(name),
    weak(weak),
    typeArgs(typeArgs) {
  //
}

bi::UnknownType::~UnknownType() {
  //
}

bi::Type* bi::UnknownType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::UnknownType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::UnknownType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::UnknownType::dispatchIsConvertible(const Type& o) const {
  return o.isConvertible(*this);
}

bi::Type* bi::UnknownType::dispatchCommon(const Type& o) const {
  return o.common(*this);
}
