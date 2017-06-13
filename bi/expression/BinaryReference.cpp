/**
 * @file
 */
#include "bi/expression/BinaryReference.hpp"

#include "bi/visitor/all.hpp"

bi::BinaryReference::BinaryReference(Expression* left, shared_ptr<Name> name,
    Expression* right, shared_ptr<Location> loc,
    const BinaryOperator* target) :
    Expression(loc),
    Named(name),
    Binary<Expression>(left, right),
    Reference<BinaryOperator>(target) {
  //
}

bi::BinaryReference::~BinaryReference() {
  //
}

bi::Expression* bi::BinaryReference::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::BinaryReference::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BinaryReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::BinaryReference::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::BinaryReference::definitely(const BinaryReference& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::BinaryReference::definitely(const BinaryOperator& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::BinaryReference::definitely(const Parameter& o) const {
  return type->definitely(*o.type);
}

bool bi::BinaryReference::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::BinaryReference::possibly(const BinaryReference& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::BinaryReference::possibly(const BinaryOperator& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::BinaryReference::possibly(const Parameter& o) const {
  return type->possibly(*o.type);
}
