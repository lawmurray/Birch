/**
 * @file
 */
#include "IdentifierType.hpp"

#include "bi/visitor/all.hpp"

template<class ObjectType>
bi::IdentifierType<ObjectType>::IdentifierType(shared_ptr<Name> name,
    shared_ptr<Location> loc, const bool assignable, const ObjectType* target) :
    Type(loc, assignable),
    Named(name),
    Reference<ObjectType>(target) {
  //
}

template<class ObjectType>
bi::IdentifierType<ObjectType>::~IdentifierType() {
  //
}

template<class ObjectType>
bi::Type* bi::IdentifierType<ObjectType>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class ObjectType>
bi::Type* bi::IdentifierType<ObjectType>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

template<class ObjectType>
void bi::IdentifierType<ObjectType>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::isBuiltin() const {
  return std::is_same<ObjectType,BasicType>::value;
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::isClass() const {
  return std::is_same<ObjectType,Class>::value;
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::isAlias() const {
  return std::is_same<ObjectType,AliasType>::value;
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::convertedDefinitely(
    const Type& o) const {
  /* pre-condition */
  assert(this->target);

  auto f = [&](const ConversionOperator* conv) {
    ///@todo Avoid transitivity here
      return conv->returnType->definitely(o);
    };
  return (!this->target->base->isEmpty() && this->target->base->definitely(o))
      || std::any_of(this->target->beginConversions(),
          this->target->endConversions(), f);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::definitely(
    const IdentifierType<Class>& o) const {
  /* pre-condition */
  assert(this->target && o.target);

  return (this->target->canonical() == o.target->canonical())
      || convertedDefinitely(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::definitely(
    const IdentifierType<AliasType>& o) const {
  /* pre-condition */
  assert(this->target && o.target);

  return (this->target->canonical() == o.target->canonical())
      || convertedDefinitely(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::definitely(
    const IdentifierType<BasicType>& o) const {
  /* pre-condition */
  assert(this->target && o.target);

  return (this->target->canonical() == o.target->canonical())
      || convertedDefinitely(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::definitely(const BracketsType& o) const {
  return convertedDefinitely(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::definitely(
    const CoroutineType& o) const {
  return convertedDefinitely(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::definitely(const EmptyType& o) const {
  return convertedDefinitely(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::definitely(const FunctionType& o) const {
  return convertedDefinitely(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::definitely(const List<Type>& o) const {
  return convertedDefinitely(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::definitely(
    const ParenthesesType& o) const {
  return definitely(*o.single);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::convertedPossibly(const Type& o) const {
  /* pre-condition */
  assert(this->target);

  auto f = [&](const ConversionOperator* conv) {
    ///@todo Avoid transitivity here
      return conv->returnType->possibly(o);
    };
  return std::any_of(this->target->beginConversions(),
      this->target->endConversions(), f);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::possibly(
    const IdentifierType<Class>& o) const {
  /* pre-condition */
  assert(this->target && o.target);

  return (this->target->canonical() == o.target->canonical())
      || convertedPossibly(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::possibly(
    const IdentifierType<BasicType>& o) const {
  /* pre-condition */
  assert(this->target && o.target);

  return (this->target->canonical() == o.target->canonical())
      || convertedPossibly(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::possibly(
    const IdentifierType<AliasType>& o) const {
  /* pre-condition */
  assert(this->target && o.target);

  return (this->target->canonical() == o.target->canonical())
      || convertedPossibly(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::possibly(const BracketsType& o) const {
  return convertedPossibly(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::possibly(const CoroutineType& o) const {
  return convertedPossibly(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::possibly(const EmptyType& o) const {
  return convertedPossibly(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::possibly(const FunctionType& o) const {
  return convertedPossibly(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::possibly(const List<Type>& o) const {
  return convertedPossibly(o);
}

template<class ObjectType>
bool bi::IdentifierType<ObjectType>::possibly(
    const ParenthesesType& o) const {
  return possibly(*o.single);
}

template class bi::IdentifierType<bi::UnknownType>;
template class bi::IdentifierType<bi::Class>;
template class bi::IdentifierType<bi::AliasType>;
template class bi::IdentifierType<bi::BasicType>;
