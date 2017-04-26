/**
 * @file
 */
#include "bi/type/ModelParameter.hpp"

#include "bi/visitor/all.hpp"

bi::ModelParameter::ModelParameter(shared_ptr<Name> name, Expression* parens,
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

bi::ModelParameter::~ModelParameter() {
  //
}

bi::Type* bi::ModelParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::ModelParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ModelParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ModelParameter::isBuiltin() const {
  if (isAlias()) {
    return base->isBuiltin();
  } else {
    return form == BUILTIN_TYPE;
  }
}

bool bi::ModelParameter::isStruct() const {
  if (isAlias()) {
    return base->isStruct();
  } else {
    return form == STRUCT_TYPE;
  }
}

bool bi::ModelParameter::isClass() const {
  if (isAlias()) {
    return base->isClass();
  } else {
    return form == CLASS_TYPE;
  }
}

bool bi::ModelParameter::isAlias() const {
  return form == ALIAS_TYPE;
}

const bi::ModelParameter* bi::ModelParameter::getBase() const {
  const ModelReference* ref =
      dynamic_cast<const ModelReference*>(base->strip());
  assert(ref && ref->target);
  return ref->target;
}

bool bi::ModelParameter::canUpcast(const ModelParameter* o) const {
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

bool bi::ModelParameter::canDowncast(const ModelParameter* o) const {
  return o->canUpcast(this);
}

bool bi::ModelParameter::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::ModelParameter::definitely(const ModelParameter& o) const {
  return parens->definitely(*o.parens) && base->definitely(*o.base)
      && (!o.assignable || assignable);
}

bool bi::ModelParameter::definitely(const ParenthesesType& o) const {
  return definitely(*o.single) && (!o.assignable || assignable);
}

bool bi::ModelParameter::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::ModelParameter::possibly(const ModelParameter& o) const {
  return parens->possibly(*o.parens) && base->possibly(*o.base)
      && (!o.assignable || assignable);
}

bool bi::ModelParameter::possibly(const ParenthesesType& o) const {
  return possibly(*o.single) && (!o.assignable || assignable);
}
