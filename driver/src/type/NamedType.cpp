/**
 * @file
 */
#include "src/type/NamedType.hpp"

#include "src/visitor/all.hpp"

birch::NamedType::NamedType(Name* name, Type* typeArgs, Location* loc) :
    Type(loc),
    Named(name),
    TypeArgumented(typeArgs) {
  //
}

birch::NamedType::NamedType(Name* name, Location* loc) :
    NamedType(name, new EmptyType(loc), loc) {
  //
}

void birch::NamedType::accept(Visitor* visitor) const {
  visitor->visit(this);
}
