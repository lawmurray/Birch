/**
 * @file
 */
#include "bi/type/TypeParameter.hpp"

#include "bi/visitor/all.hpp"

bi::TypeParameter::TypeParameter(shared_ptr<Name> name, Expression* parens,
    Type* base, Expression* braces, const TypeForm form,
    shared_ptr<Location> loc, const bool assignable) :
    Type(loc, assignable),
    Named(name),
    Parenthesised(parens),
    Based(base),
    Braced(braces),
    form(form) {
  //
}

bi::TypeParameter::~TypeParameter() {
  //
}

bi::Type* bi::TypeParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::TypeParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::TypeParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::TypeParameter::isBuiltin() const {
  if (isAlias()) {
    return base->isBuiltin();
  } else {
    return form == BUILTIN_TYPE;
  }
}

bool bi::TypeParameter::isStruct() const {
  if (isAlias()) {
    return base->isStruct();
  } else {
    return form == STRUCT_TYPE;
  }
}

bool bi::TypeParameter::isClass() const {
  if (isAlias()) {
    return base->isClass();
  } else {
    return form == CLASS_TYPE;
  }
}

bool bi::TypeParameter::isAlias() const {
  return form == ALIAS_TYPE;
}

const bi::TypeParameter* bi::TypeParameter::getBase() const {
  const TypeReference* ref =
      dynamic_cast<const TypeReference*>(base->strip());
  assert(ref && ref->target);
  return ref->target;
}

bool bi::TypeParameter::canUpcast(const TypeParameter* o) const {
  if (o->isAlias()) {
    return canUpcast(o->getBase());
  } else if (isAlias()) {
    return getBase()->canUpcast(o);
  } else if (!base->isEmpty()) {
    return this == o || getBase()->canUpcast(o);
  } else {
    return this == o;
  }
}

bool bi::TypeParameter::canDowncast(const TypeParameter* o) const {
  return o->canUpcast(this);
}

bool bi::TypeParameter::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::TypeParameter::definitely(const TypeParameter& o) const {
  return parens->definitely(*o.parens) && base->definitely(*o.base)
      && (!o.assignable || assignable);
}

bool bi::TypeParameter::definitely(const ParenthesesType& o) const {
  return definitely(*o.single) && (!o.assignable || assignable);
}

bool bi::TypeParameter::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::TypeParameter::possibly(const TypeParameter& o) const {
  return parens->possibly(*o.parens) && base->possibly(*o.base)
      && (!o.assignable || assignable);
}

bool bi::TypeParameter::possibly(const ParenthesesType& o) const {
  return possibly(*o.single) && (!o.assignable || assignable);
}
