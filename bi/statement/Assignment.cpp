/**
 * @file
 */
#include "bi/statement/Assignment.hpp"

#include "bi/visitor/all.hpp"

bi::Assignment::Assignment(Expression* left,
    shared_ptr<Name> name, Expression* right, shared_ptr<Location> loc,
    const AssignmentOperator* target) :
    Statement(loc),
    Named(name),
    Binary<Expression>(left, right),
    Reference<AssignmentOperator>(target) {
  //
}

bi::Assignment::~Assignment() {
  //
}

bi::Statement* bi::Assignment::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Assignment::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Assignment::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::Assignment::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::Assignment::definitely(const Assignment& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::Assignment::definitely(const AssignmentOperator& o) const {
  return right->definitely(*o.single);
}

bool bi::Assignment::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::Assignment::possibly(const Assignment& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::Assignment::possibly(const AssignmentOperator& o) const {
  return right->possibly(*o.single);
}
