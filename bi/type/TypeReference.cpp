/**
 * @file
 */
#include "bi/type/TypeReference.hpp"

#include "bi/visitor/all.hpp"

bi::TypeReference::TypeReference(shared_ptr<Name> name,
    shared_ptr<Location> loc, const bool assignable,
    const TypeParameter* target) :
    Type(loc, assignable),
    Named(name),
    Reference(target) {
  //
}

bi::TypeReference::TypeReference(const TypeParameter* target) :
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

bool bi::TypeReference::convertedDefinitely(const Type& o) const {
  /* pre-condition */
  assert(target);

  auto f = [&](const ConversionParameter* conv) {
    ///@todo Avoid transitivity here
      return conv->type->definitely(o);
    };
  return (!target->base->isEmpty() && target->base->definitely(o))
      || std::any_of(target->beginConversions(), target->endConversions(), f);
}

bool bi::TypeReference::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::TypeReference::definitely(const BracketsType& o) const {
  return convertedDefinitely(o);
}

bool bi::TypeReference::definitely(const EmptyType& o) const {
  return convertedDefinitely(o);
}

bool bi::TypeReference::definitely(const LambdaType& o) const {
  return convertedDefinitely(o);
}

bool bi::TypeReference::definitely(const List<Type>& o) const {
  return convertedDefinitely(o);
}

bool bi::TypeReference::definitely(const TypeParameter& o) const {
  return true;
}

bool bi::TypeReference::definitely(const TypeReference& o) const {
  /* pre-condition */
  assert(target && o.target);

  return (target->canonical() == o.target->canonical())
      || convertedDefinitely(o);
}

bool bi::TypeReference::definitely(const ParenthesesType& o) const {
  return definitely(*o.single);
}

bool bi::TypeReference::convertedPossibly(const Type& o) const {
  /* pre-condition */
  assert(target);

  auto f = [&](const ConversionParameter* conv) {
    ///@todo Avoid transitivity here
      return conv->type->possibly(o);
    };
  return std::any_of(target->beginConversions(), target->endConversions(), f);
}

bool bi::TypeReference::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::TypeReference::possibly(const BracketsType& o) const {
  return convertedPossibly(o);
}

bool bi::TypeReference::possibly(const EmptyType& o) const {
  return convertedPossibly(o);
}

bool bi::TypeReference::possibly(const LambdaType& o) const {
  return convertedPossibly(o);
}

bool bi::TypeReference::possibly(const List<Type>& o) const {
  return convertedPossibly(o);
}

bool bi::TypeReference::possibly(const TypeParameter& o) const {
  return true;
}

bool bi::TypeReference::possibly(const TypeReference& o) const {
  /* pre-condition */
  assert(target && o.target);

  return (target->canonical() == o.target->canonical())
      || convertedPossibly(o);
}

bool bi::TypeReference::possibly(const ParenthesesType& o) const {
  return possibly(*o.single);
}
