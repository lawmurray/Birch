/**
 * @file
 */
#include "bi/type/NamedType.hpp"

#include "bi/visitor/all.hpp"

bi::NamedType::NamedType(const bool weak, Name* name, Type* typeArgs,
    Location* loc) :
    Type(loc),
    Named(name),
    TypeArgumented(typeArgs),
    weak(weak) {
  //
}

bi::NamedType::NamedType(const bool weak, Name* name, Location* loc) :
    NamedType(weak, name, new EmptyType(loc), loc) {
  //
}

bi::NamedType::NamedType(Name* name, Location* loc) :
    NamedType(false, name, new EmptyType(loc), loc) {
  //
}

bi::NamedType::~NamedType() {
  //
}

bi::Type* bi::NamedType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::NamedType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::NamedType::accept(Visitor* visitor) const {
  visitor->visit(this);
}
