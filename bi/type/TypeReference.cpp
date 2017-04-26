/**
 * @file
 */
#include "bi/type/TypeReference.hpp"

#include "bi/visitor/all.hpp"

bi::TypeReference::TypeReference(shared_ptr<Name> name, Expression* parens,
    shared_ptr<Location> loc, const bool assignable, TypeParameter* target) :
    Type(loc, assignable),
    Named(name),
    Parenthesised(parens),
    Reference(target) {
  //
}

bi::TypeReference::TypeReference(TypeParameter* target) :
    Named(target->name),
    Reference(target) {
  //
}

bi::TypeReference::~TypeReference() {
  //
}

bi::Type* bi::TypeReference::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::TypeReference::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::TypeReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::TypeReference::isBuiltin() const {
  /* pre-condition */
  assert(target);

  return target->isBuiltin();
}

bool bi::TypeReference::isStruct() const {
  /* pre-condition */
  assert(target);

  return target->isStruct();
}

bool bi::TypeReference::isClass() const {
  /* pre-condition */
  assert(target);

  return target->isClass();
}

bool bi::TypeReference::isAlias() const {
  /* pre-condition */
  assert(target);

  return target->isAlias();
}

bool bi::TypeReference::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::TypeReference::definitely(const TypeParameter& o) const {
  return true;
}

bool bi::TypeReference::definitely(const TypeReference& o) const {
  return target->canUpcast(o.target) && (!o.assignable || assignable);
}

bool bi::TypeReference::definitely(const ParenthesesType& o) const {
  return definitely(*o.single) && (!o.assignable || assignable);
}

bool bi::TypeReference::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::TypeReference::possibly(const TypeParameter& o) const {
  return true;
}

bool bi::TypeReference::possibly(const TypeReference& o) const {
  /* pre-condition */
  assert(target && o.target);

  return target->canDowncast(o.target) && (!o.assignable || assignable);
}

bool bi::TypeReference::possibly(const ParenthesesType& o) const {
  return possibly(*o.single) && (!o.assignable || assignable);
}
