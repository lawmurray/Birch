/**
 * @file
 */
#include "bi/expression/BinaryCall.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::OverloadedCall<bi::BinaryOperator>::OverloadedCall(Expression* left,
    shared_ptr<Name> name, Expression* right, shared_ptr<Location> loc,
    BinaryOperator* target) :
    Expression(loc),
    Named(name),
    Binary<Expression>(left, right),
    Reference<BinaryOperator>(target) {
  //
}

bi::OverloadedCall<bi::BinaryOperator>::~OverloadedCall() {
  //
}

bi::Expression* bi::OverloadedCall<bi::BinaryOperator>::accept(
    Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::OverloadedCall<bi::BinaryOperator>::accept(
    Modifier* visitor) {
  return visitor->modify(this);
}

void bi::OverloadedCall<bi::BinaryOperator>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::OverloadedCall<bi::BinaryOperator>::dispatchDefinitely(
    const Expression& o) const {
  return o.definitely(*this);
}

bool bi::OverloadedCall<bi::BinaryOperator>::definitely(
    const OverloadedCall<BinaryOperator>& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::OverloadedCall<bi::BinaryOperator>::definitely(
    const BinaryOperator& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::OverloadedCall<bi::BinaryOperator>::definitely(
    const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::OverloadedCall<bi::BinaryOperator>::dispatchPossibly(
    const Expression& o) const {
  return o.possibly(*this);
}

bool bi::OverloadedCall<bi::BinaryOperator>::possibly(
    const OverloadedCall<BinaryOperator>& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::OverloadedCall<bi::BinaryOperator>::possibly(
    const BinaryOperator& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::OverloadedCall<bi::BinaryOperator>::possibly(
    const Parameter& o) const {
  return type->possibly(*o.type);
}
