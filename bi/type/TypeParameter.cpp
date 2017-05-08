/**
 * @file
 */
#include "bi/type/TypeParameter.hpp"

#include "bi/visitor/all.hpp"

bi::TypeParameter::TypeParameter(shared_ptr<Name> name, Expression* parens,
    Type* base, Expression* baseParens, Expression* braces,
    const TypeForm form, shared_ptr<Location> loc, const bool assignable) :
    Type(loc, assignable),
    Named(name),
    Parenthesised(parens),
    Based(base, baseParens),
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

const bi::TypeParameter* bi::TypeParameter::canonical() const {
  if (isAlias()) {
    return super()->canonical();
  } else {
    return this;
  }
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
  return parens->definitely(*o.parens);
}

bool bi::TypeParameter::definitely(const ParenthesesType& o) const {
  return definitely(*o.single);
}

bool bi::TypeParameter::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::TypeParameter::possibly(const TypeParameter& o) const {
  return parens->possibly(*o.parens);
}

bool bi::TypeParameter::possibly(const ParenthesesType& o) const {
  return possibly(*o.single);
}
