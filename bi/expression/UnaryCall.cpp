/**
 * @file
 */
#include "bi/expression/Call.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::OverloadedCall<bi::UnaryOperator>::OverloadedCall(shared_ptr<Name> name,
    Expression* single, shared_ptr<Location> loc, const UnaryOperator* target) :
    Expression(loc),
    Named(name),
    Unary<Expression>(single),
    Reference<UnaryOperator>(target) {
  //
}

bi::OverloadedCall<bi::UnaryOperator>::~OverloadedCall() {
  //
}

bi::Expression* bi::OverloadedCall<bi::UnaryOperator>::accept(
    Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::OverloadedCall<bi::UnaryOperator>::accept(
    Modifier* visitor) {
  return visitor->modify(this);
}

void bi::OverloadedCall<bi::UnaryOperator>::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::OverloadedCall<bi::UnaryOperator>::dispatchDefinitely(
    const Expression& o) const {
  return o.definitely(*this);
}

bool bi::OverloadedCall<bi::UnaryOperator>::definitely(
    const OverloadedCall<UnaryOperator>& o) const {
  return single->definitely(*o.single);
}

bool bi::OverloadedCall<bi::UnaryOperator>::definitely(
    const UnaryOperator& o) const {
  return single->definitely(*o.single);
}

bool bi::OverloadedCall<bi::UnaryOperator>::definitely(
    const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::OverloadedCall<bi::UnaryOperator>::dispatchPossibly(
    const Expression& o) const {
  return o.possibly(*this);
}

bool bi::OverloadedCall<bi::UnaryOperator>::possibly(
    const OverloadedCall<UnaryOperator>& o) const {
  return single->possibly(*o.single);
}

bool bi::OverloadedCall<bi::UnaryOperator>::possibly(
    const UnaryOperator& o) const {
  return single->possibly(*o.single);
}

bool bi::OverloadedCall<bi::UnaryOperator>::possibly(
    const Parameter& o) const {
  return type->possibly(*o.type);
}
