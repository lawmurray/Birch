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

const bi::TypeParameter* bi::TypeParameter::super() const {
  const TypeReference* result = dynamic_cast<const TypeReference*>(base.get());
  if (result) {
    return result->target;
  } else {
    return nullptr;
  }
}

bool bi::TypeParameter::derivedFrom(const TypeParameter* o) const {
  if (isAlias()) {
    return super()->derivedFrom(o);
  } else if (o->isAlias()) {
    return derivedFrom(o->super());
  } else if (this == o) {
    return true;
  } else {
    return super() && super()->derivedFrom(o);
  }
}

bool bi::TypeParameter::convertibleTo(const TypeParameter* o) const {
  auto f =
      [&](const ConversionParameter* conv) {
        const TypeReference* ref = dynamic_cast<const TypeReference*>(conv->type.get());
        return ref && ref->target && ref->target->derivedFrom(o);
      };
  return derivedFrom(o)
      || std::any_of(beginConversions(), endConversions(), f);
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
