/**
 * @file
 */
#include "bi/expression/BinaryCall.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::Call<bi::BinaryOperator>::Call(Expression* left, shared_ptr<Name> name,
    Expression* right, shared_ptr<Location> loc, const BinaryOperator* target) :
    Expression(loc),
    Named(name),
    Binary<Expression>(left, right),
    Reference<BinaryOperator>(target) {
  //
}

bi::Call<bi::BinaryOperator>::~Call() {
  //
}

bi::Expression* bi::Call<bi::BinaryOperator>::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Call<bi::BinaryOperator>::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Call<bi::BinaryOperator>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Call<bi::BinaryOperator>::dispatchDefinitely(
    const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Call<bi::BinaryOperator>::definitely(
    const Call<BinaryOperator>& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::Call<bi::BinaryOperator>::definitely(const BinaryOperator& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::Call<bi::BinaryOperator>::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::Call<bi::BinaryOperator>::dispatchPossibly(
    const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Call<bi::BinaryOperator>::possibly(
    const Call<BinaryOperator>& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::Call<bi::BinaryOperator>::possibly(const BinaryOperator& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::Call<bi::BinaryOperator>::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}
