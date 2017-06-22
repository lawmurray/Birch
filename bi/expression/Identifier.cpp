/**
 * @file
 */
#include "bi/expression/Identifier.hpp"

#include "bi/visitor/all.hpp"

template<class ObjectType>
bi::Identifier<ObjectType>::Identifier(shared_ptr<Name> name,
    Expression* parens, shared_ptr<Location> loc, const ObjectType* target) :
    Expression(loc),
    Named(name),
    Parenthesised(parens),
    Reference<ObjectType>(target) {
  //
}

template<class ObjectType>
bi::Identifier<ObjectType>::~Identifier() {
  //
}

template<class ObjectType>
bi::Expression* bi::Identifier<ObjectType>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

template<class ObjectType>
bi::Expression* bi::Identifier<ObjectType>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

template<class ObjectType>
void bi::Identifier<ObjectType>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::dispatchDefinitely(
    const Expression& o) const {
  return o.definitely(*this);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::definitely(
    const Identifier<ObjectType>& o) const {
  return parens->definitely(*o.parens);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::definitely(const Parameter& o) const {
  return !this->target || type->definitely(*o.type);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::definitely(const GlobalVariable& o) const {
  return !this->target;
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::definitely(const LocalVariable& o) const {
  return !this->target;
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::definitely(const MemberVariable& o) const {
  return !this->target;
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::definitely(const Function& o) const {
  return !this->target && parens->definitely(*o.parens);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::definitely(const Coroutine& o) const {
  return !this->target && parens->definitely(*o.parens);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::definitely(const MemberFunction& o) const {
  return !this->target && parens->definitely(*o.parens);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::possibly(
    const Identifier<ObjectType>& o) const {
  return parens->possibly(*o.parens);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::possibly(const Parameter& o) const {
  return !this->target || type->possibly(*o.type);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::possibly(const GlobalVariable& o) const {
  return !this->target;
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::possibly(const LocalVariable& o) const {
  return !this->target;
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::possibly(const MemberVariable& o) const {
  return !this->target;
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::possibly(const Function& o) const {
  return !this->target && parens->possibly(*o.parens);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::possibly(const Coroutine& o) const {
  return !this->target && parens->possibly(*o.parens);
}

template<class ObjectType>
bool bi::Identifier<ObjectType>::possibly(const MemberFunction& o) const {
  return !this->target && parens->possibly(*o.parens);
}

bi::Identifier<bi::BinaryOperator>::Identifier(Expression* left,
    shared_ptr<Name> name, Expression* right, shared_ptr<Location> loc,
    const BinaryOperator* target) :
    Expression(loc),
    Named(name),
    Binary<Expression>(left, right),
    Reference<BinaryOperator>(target) {
  //
}

bi::Identifier<bi::BinaryOperator>::~Identifier() {
  //
}

bi::Expression* bi::Identifier<bi::BinaryOperator>::accept(
    Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Identifier<bi::BinaryOperator>::accept(
    Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Identifier<bi::BinaryOperator>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Identifier<bi::BinaryOperator>::dispatchDefinitely(
    const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Identifier<bi::BinaryOperator>::definitely(
    const Identifier<BinaryOperator>& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::Identifier<bi::BinaryOperator>::definitely(
    const BinaryOperator& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::Identifier<bi::BinaryOperator>::definitely(
    const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::Identifier<bi::BinaryOperator>::dispatchPossibly(
    const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Identifier<bi::BinaryOperator>::possibly(
    const Identifier<BinaryOperator>& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::Identifier<bi::BinaryOperator>::possibly(
    const BinaryOperator& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::Identifier<bi::BinaryOperator>::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}

bi::Identifier<bi::UnaryOperator>::Identifier(shared_ptr<Name> name,
    Expression* single, shared_ptr<Location> loc, const UnaryOperator* target) :
    Expression(loc),
    Named(name),
    Unary<Expression>(single),
    Reference<UnaryOperator>(target) {
  //
}

bi::Identifier<bi::UnaryOperator>::~Identifier<UnaryOperator>() {
  //
}

bi::Expression* bi::Identifier<bi::UnaryOperator>::accept(
    Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Identifier<bi::UnaryOperator>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Identifier<bi::UnaryOperator>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Identifier<bi::UnaryOperator>::dispatchDefinitely(
    const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Identifier<bi::UnaryOperator>::definitely(
    const Identifier<UnaryOperator>& o) const {
  return single->definitely(*o.single);
}

bool bi::Identifier<bi::UnaryOperator>::definitely(
    const UnaryOperator& o) const {
  return single->definitely(*o.single);
}

bool bi::Identifier<bi::UnaryOperator>::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::Identifier<bi::UnaryOperator>::dispatchPossibly(
    const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Identifier<bi::UnaryOperator>::possibly(
    const Identifier<UnaryOperator>& o) const {
  return single->possibly(*o.single);
}

bool bi::Identifier<bi::UnaryOperator>::possibly(
    const UnaryOperator& o) const {
  return single->possibly(*o.single);
}

bool bi::Identifier<bi::UnaryOperator>::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}

template class bi::Identifier<bi::Unknown>;
template class bi::Identifier<bi::Parameter>;
template class bi::Identifier<bi::MemberParameter>;
template class bi::Identifier<bi::GlobalVariable>;
template class bi::Identifier<bi::LocalVariable>;
template class bi::Identifier<bi::MemberVariable>;
template class bi::Identifier<bi::Function>;
template class bi::Identifier<bi::Coroutine>;
template class bi::Identifier<bi::MemberFunction>;
template class bi::Identifier<bi::BinaryOperator>;
template class bi::Identifier<bi::UnaryOperator>;
