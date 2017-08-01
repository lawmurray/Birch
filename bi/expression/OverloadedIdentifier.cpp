/**
 * @file
 */
#include "bi/expression/OverloadedIdentifier.hpp"

#include "bi/visitor/all.hpp"

template<class ObjectType>
bi::OverloadedIdentifier<ObjectType>::OverloadedIdentifier(
    shared_ptr<Name> name, shared_ptr<Location> loc,
    Overloaded<ObjectType>* target) :
    Expression(loc),
    Named(name),
    Reference<Overloaded<ObjectType>>(target) {
  //
}

template<class ObjectType>
bi::OverloadedIdentifier<ObjectType>::~OverloadedIdentifier() {
  //
}

template<class ObjectType>
bool bi::OverloadedIdentifier<ObjectType>::isOverloaded() const {
  return true;
}

template<class ObjectType>
bi::Expression* bi::OverloadedIdentifier<ObjectType>::accept(
    Cloner* visitor) const {
  return visitor->clone(this);
}

template<class ObjectType>
bi::Expression* bi::OverloadedIdentifier<ObjectType>::accept(
    Modifier* visitor) {
  return visitor->modify(this);
}

template<class ObjectType>
void bi::OverloadedIdentifier<ObjectType>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template class bi::OverloadedIdentifier<bi::Function>;
template class bi::OverloadedIdentifier<bi::Coroutine>;
template class bi::OverloadedIdentifier<bi::MemberFunction>;
template class bi::OverloadedIdentifier<bi::MemberCoroutine>;
template class bi::OverloadedIdentifier<bi::BinaryOperator>;
template class bi::OverloadedIdentifier<bi::UnaryOperator>;
