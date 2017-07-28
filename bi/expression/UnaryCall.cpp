/**
 * @file
 */
#include "bi/expression/Call.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::UnaryCall::UnaryCall(shared_ptr<Name> name, Expression* single,
    shared_ptr<Location> loc, const UnaryOperator* target) :
    Expression(loc),
    Named(name),
    Unary<Expression>(single),
    Reference<UnaryOperator>(target) {
  //
}

bi::UnaryCall::~UnaryCall() {
  //
}

bi::Expression* bi::UnaryCall::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::UnaryCall::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::UnaryCall::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::UnaryCall::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::UnaryCall::definitely(const UnaryCall& o) const {
  return single->definitely(*o.single);
}

bool bi::UnaryCall::definitely(const UnaryOperator& o) const {
  return single->definitely(*o.single);
}

bool bi::UnaryCall::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::UnaryCall::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::UnaryCall::possibly(const UnaryCall& o) const {
  return single->possibly(*o.single);
}

bool bi::UnaryCall::possibly(const UnaryOperator& o) const {
  return single->possibly(*o.single);
}

bool bi::UnaryCall::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}
