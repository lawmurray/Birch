/**
 * @file
 */
#include "OverloadedCall.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

template<class ObjectType>
bi::OverloadedCall<ObjectType>::OverloadedCall(Expression* single,
    Expression* parens, shared_ptr<Location> loc, const ObjectType* target) :
    Expression(loc),
    Unary<Expression>(single),
    Parenthesised(parens),
    Reference<ObjectType>(target) {
  //
}

template<class ObjectType>
bi::OverloadedCall<ObjectType>::~OverloadedCall() {
  //
}

template<class ObjectType>
bi::Expression* bi::OverloadedCall<ObjectType>::accept(
    Cloner* visitor) const {
  return visitor->clone(this);
}

template<class ObjectType>
bi::Expression* bi::OverloadedCall<ObjectType>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

template<class ObjectType>
void bi::OverloadedCall<ObjectType>::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

template<class ObjectType>
bool bi::OverloadedCall<ObjectType>::dispatchDefinitely(
    const Expression& o) const {
  return o.definitely(*this);
}

template<class ObjectType>
bool bi::OverloadedCall<ObjectType>::definitely(
    const OverloadedCall<ObjectType>& o) const {
  return single->definitely(*o.single) && parens->definitely(*o.parens);
}

template<class ObjectType>
bool bi::OverloadedCall<ObjectType>::definitely(const ObjectType& o) const {
  return parens->definitely(*o.parens);
}

template<class ObjectType>
bool bi::OverloadedCall<ObjectType>::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

template<class ObjectType>
bool bi::OverloadedCall<ObjectType>::dispatchPossibly(
    const Expression& o) const {
  return o.possibly(*this);
}

template<class ObjectType>
bool bi::OverloadedCall<ObjectType>::possibly(
    const OverloadedCall<ObjectType>& o) const {
  return single->possibly(*o.single) && parens->possibly(*o.parens);
}

template<class ObjectType>
bool bi::OverloadedCall<ObjectType>::possibly(const ObjectType& o) const {
  return parens->possibly(*o.parens);
}

template<class ObjectType>
bool bi::OverloadedCall<ObjectType>::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}

template class bi::OverloadedCall<bi::Function>;
template class bi::OverloadedCall<bi::Coroutine>;
template class bi::OverloadedCall<bi::MemberFunction>;
template class bi::OverloadedCall<bi::MemberCoroutine>;
