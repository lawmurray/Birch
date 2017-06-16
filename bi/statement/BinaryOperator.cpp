/**
 * @file
 */
#include "bi/statement/BinaryOperator.hpp"

#include "bi/visitor/all.hpp"

bi::BinaryOperator::BinaryOperator(Expression* left, shared_ptr<Name> name,
    Expression* right, Type* returnType, Statement* braces,
    shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Binary(left, right),
    ReturnTyped(returnType),
    Braced(braces) {
  //
}

bi::BinaryOperator::~BinaryOperator() {
  //
}

bi::Statement* bi::BinaryOperator::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::BinaryOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BinaryOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::BinaryOperator::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::BinaryOperator::definitely(const BinaryOperator& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::BinaryOperator::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::BinaryOperator::possibly(const BinaryOperator& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}
