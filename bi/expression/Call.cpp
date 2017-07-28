/**
 * @file
 */
#include "bi/expression/Call.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

template<class ObjectType>
bi::Call<ObjectType>::Call(Expression* single, Expression* parens,
    shared_ptr<Location> loc, const ObjectType* target) :
    Expression(loc),
    Unary<Expression>(single),
    Parenthesised(parens),
    Reference<ObjectType>(target) {
  //
}

template<class ObjectType>
bi::Call<ObjectType>::~Call() {
  //
}

template<class ObjectType>
bi::Expression* bi::Call<ObjectType>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class ObjectType>
bi::Expression* bi::Call<ObjectType>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

template<class ObjectType>
void bi::Call<ObjectType>::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

template<class ObjectType>
bool bi::Call<ObjectType>::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

template<class ObjectType>
bool bi::Call<ObjectType>::definitely(const Call<ObjectType>& o) const {
  return single->definitely(*o.single) && parens->definitely(*o.parens);
}

template<class ObjectType>
bool bi::Call<ObjectType>::definitely(const ObjectType& o) const {
  return parens->definitely(*o.parens);
}

template<class ObjectType>
bool bi::Call<ObjectType>::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

template<class ObjectType>
bool bi::Call<ObjectType>::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

template<class ObjectType>
bool bi::Call<ObjectType>::possibly(const Call<ObjectType>& o) const {
  return single->possibly(*o.single) && parens->possibly(*o.parens);
}

template<class ObjectType>
bool bi::Call<ObjectType>::possibly(const ObjectType& o) const {
  return parens->possibly(*o.parens);
}

template<class ObjectType>
bool bi::Call<ObjectType>::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}

bi::Call<bi::Expression>::Call(Expression* single, Expression* parens,
    shared_ptr<Location> loc, const Expression* target) :
    Expression(loc),
    Unary<Expression>(single),
    Parenthesised(parens),
    Reference<Expression>(target) {
  //
}

bi::Call<bi::Expression>::~Call() {
  //
}

bi::Expression* bi::Call<bi::Expression>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Call<bi::Expression>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Call<bi::Expression>::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::Call<bi::Expression>::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Call<bi::Expression>::definitely(const Call<Expression>& o) const {
  return single->definitely(*o.single) && parens->definitely(*o.parens);
}

bool bi::Call<bi::Expression>::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::Call<bi::Expression>::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Call<bi::Expression>::possibly(const Call<Expression>& o) const {
  return single->possibly(*o.single) && parens->possibly(*o.parens);
}

bool bi::Call<bi::Expression>::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}

//template class bi::Call<bi::Function>;
//template class bi::Call<bi::Coroutine>;
//template class bi::Call<bi::MemberFunction>;
//template class bi::Call<bi::MemberCoroutine>;
