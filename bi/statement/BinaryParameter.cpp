/**
 * @file
 */
#include "bi/statement/BinaryParameter.hpp"

#include "bi/visitor/all.hpp"

bi::BinaryParameter::BinaryParameter(Expression* left, shared_ptr<Name> name,
    Expression* right, Type* type, Expression* braces,
    shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Binary(left, right),
    Typed(type),
    Braced(braces) {
  //
}

bi::BinaryParameter::~BinaryParameter() {
  //
}

bi::Statement* bi::BinaryParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::BinaryParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BinaryParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::BinaryParameter::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::BinaryParameter::definitely(const BinaryParameter& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::BinaryParameter::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::BinaryParameter::possibly(const BinaryParameter& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}
