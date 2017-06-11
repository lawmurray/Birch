/**
 * @file
 */
#include "bi/statement/AssignmentReference.hpp"

#include "bi/visitor/all.hpp"

bi::AssignmentReference::AssignmentReference(Expression* left,
    shared_ptr<Name> name, Expression* right, shared_ptr<Location> loc,
    const AssignmentParameter* target) :
    Statement(loc),
    Named(name),
    Binary<Expression>(left, right),
    Reference<AssignmentParameter>(target) {
  //
}

bi::AssignmentReference::~AssignmentReference() {
  //
}

bi::Statement* bi::AssignmentReference::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::AssignmentReference::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::AssignmentReference::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::AssignmentReference::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::AssignmentReference::definitely(const AssignmentReference& o) const {
  return left->definitely(*o.left) && right->definitely(*o.right);
}

bool bi::AssignmentReference::definitely(const AssignmentParameter& o) const {
  return right->definitely(*o.single);
}

bool bi::AssignmentReference::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::AssignmentReference::possibly(const AssignmentReference& o) const {
  return left->possibly(*o.left) && right->possibly(*o.right);
}

bool bi::AssignmentReference::possibly(const AssignmentParameter& o) const {
  return right->possibly(*o.single);
}
