/**
 * @file
 */
#include "src/statement/AssignmentOperator.hpp"

#include "src/visitor/all.hpp"

birch::AssignmentOperator::AssignmentOperator(const Annotation annotation,
    Expression* single, Statement* braces, Location* loc) :
    Statement(loc),
    Annotated(annotation),
    Single(single),
    Braced(braces) {
  //
}

void birch::AssignmentOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
