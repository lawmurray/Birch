/**
 * @file
 */
#include "bi/statement/AssignmentOperator.hpp"

#include "bi/visitor/all.hpp"

bi::AssignmentOperator::AssignmentOperator(shared_ptr<Name> name,
    Expression* single, Statement* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Unary(single),
    Braced(braces) {
  //
}

bi::AssignmentOperator::~AssignmentOperator() {
  //
}

bi::Statement* bi::AssignmentOperator::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::AssignmentOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::AssignmentOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::AssignmentOperator::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::AssignmentOperator::definitely(const AssignmentOperator& o) const {
  return single->definitely(*o.single);
}

bool bi::AssignmentOperator::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::AssignmentOperator::possibly(const AssignmentOperator& o) const {
  return single->possibly(*o.single);
}
