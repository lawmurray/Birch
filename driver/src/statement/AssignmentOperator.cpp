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
    Scoped(LOCAL_SCOPE),
    Braced(braces) {
  //
}

birch::AssignmentOperator::~AssignmentOperator() {
  //
}

birch::Statement* birch::AssignmentOperator::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Statement* birch::AssignmentOperator::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::AssignmentOperator::accept(Visitor* visitor) const {
  visitor->visit(this);
}
