/**
 * @file
 */
#include "bi/expression/OverloadedIdentifier.hpp"

#include "bi/visitor/all.hpp"

template<class ObjectType>
bi::OverloadedIdentifier<ObjectType>::OverloadedIdentifier(
    shared_ptr<Name> name, shared_ptr<Location> loc,
    const Overloaded<ObjectType>* target) :
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

template<class ObjectType>
bool bi::OverloadedIdentifier<ObjectType>::dispatchDefinitely(
    const Expression& o) const {
  return o.definitely(*this);
}

template<class ObjectType>
bool bi::OverloadedIdentifier<ObjectType>::definitely(
    const OverloadedIdentifier<ObjectType>& o) const {
  return true;
}

template<class ObjectType>
bool bi::OverloadedIdentifier<ObjectType>::definitely(
    const Parameter& o) const {
  return type->definitely(*o.type);
}

template<class ObjectType>
bool bi::OverloadedIdentifier<ObjectType>::dispatchPossibly(
    const Expression& o) const {
  return o.possibly(*this);
}

template<class ObjectType>
bool bi::OverloadedIdentifier<ObjectType>::possibly(
    const OverloadedIdentifier<ObjectType>& o) const {
  return true;
}

template<class ObjectType>
bool bi::OverloadedIdentifier<ObjectType>::possibly(
    const Parameter& o) const {
  return type->possibly(*o.type);
}

template class bi::OverloadedIdentifier<bi::Function>;
template class bi::OverloadedIdentifier<bi::Coroutine>;
template class bi::OverloadedIdentifier<bi::MemberFunction>;
template class bi::OverloadedIdentifier<bi::MemberCoroutine>;
