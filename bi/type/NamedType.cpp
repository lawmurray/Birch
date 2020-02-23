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
    weak(weak),
    category(UNKNOWN_TYPE),
    number(0) {
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

bi::NamedType::NamedType(Class* target, Location* loc) :
    NamedType(false, target->name, target->createArguments(), loc) {
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

bool bi::NamedType::isBasic() const {
  return category == BASIC_TYPE;
}

bool bi::NamedType::isClass() const {
  return category == CLASS_TYPE;
}

bool bi::NamedType::isGeneric() const {
  return category == GENERIC_TYPE;
}

bool bi::NamedType::isWeak() const {
  return weak;
}
