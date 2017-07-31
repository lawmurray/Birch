/**
 * @file
 */
#include "bi/expression/BinaryCall.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::BinaryCall::BinaryCall(Expression* left, shared_ptr<Name> name,
    Expression* right, shared_ptr<Location> loc) :
    Expression(loc),
    Named(name),
    Binary<Expression>(left, right) {
  //
}

bi::BinaryCall::~BinaryCall() {
  //
}

bi::Expression* bi::BinaryCall::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::BinaryCall::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BinaryCall::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::BinaryCall::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::BinaryCall::definitely(const BinaryCall& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::BinaryCall::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::BinaryCall::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::BinaryCall::possibly(const BinaryCall& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::BinaryCall::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}
