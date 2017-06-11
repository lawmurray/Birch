/**
 * @file
 */
#include "bi/expression/BinaryParameter.hpp"

#include "bi/visitor/all.hpp"

bi::BinaryParameter::BinaryParameter(Expression* left, shared_ptr<Name> name,
    Expression* right, Type* type, Expression* braces,
    shared_ptr<Location> loc) :
    Expression(type, loc),
    Named(name),
    Binary(left, right),
    Braced(braces) {
  //
}

bi::BinaryParameter::~BinaryParameter() {
  //
}

bi::Expression* bi::BinaryParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::BinaryParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BinaryParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::BinaryParameter::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::BinaryParameter::definitely(const BinaryParameter& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::BinaryParameter::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::BinaryParameter::possibly(const BinaryParameter& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}
