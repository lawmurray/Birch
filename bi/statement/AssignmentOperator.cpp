/**
 * @file
 */
#include "bi/statement/AssignmentOperator.hpp"

#include "bi/visitor/all.hpp"

bi::AssignmentOperator::AssignmentOperator(Name* name,
    Expression* single, Statement* braces, Location* loc) :
    Statement(loc),
    Named(name),
    Single(single),
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
