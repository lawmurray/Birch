/**
 * @file
 */
#include "src/type/NamedType.hpp"

#include "src/visitor/all.hpp"

birch::NamedType::NamedType(Name* name, Type* typeArgs, Location* loc) :
    Type(loc),
    Named(name),
    TypeArgumented(typeArgs),
    category(UNKNOWN_TYPE),
    number(0) {
  //
}

birch::NamedType::NamedType(Name* name, Location* loc) :
    NamedType(name, new EmptyType(loc), loc) {
  //
}

birch::Type* birch::NamedType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::NamedType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool birch::NamedType::isBasic() const {
  return category == BASIC_TYPE;
}

bool birch::NamedType::isStruct() const {
  return category == STRUCT_TYPE;
}

bool birch::NamedType::isClass() const {
  return category == CLASS_TYPE;
}

bool birch::NamedType::isGeneric() const {
  return category == GENERIC_TYPE;
}

bool birch::NamedType::isValue() const {
  return isBasic();
}
