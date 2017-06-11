/**
 * @file
 */
#include "bi/statement/AssignmentParameter.hpp"

#include "bi/visitor/all.hpp"

bi::AssignmentParameter::AssignmentParameter(shared_ptr<Name> name,
    Expression* single, Expression* braces, shared_ptr<Location> loc) :
    Statement(loc),
    Named(name),
    Unary(single),
    Braced(braces) {
  //
}

bi::AssignmentParameter::~AssignmentParameter() {
  //
}

bi::Statement* bi::AssignmentParameter::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::AssignmentParameter::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::AssignmentParameter::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::AssignmentParameter::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::AssignmentParameter::definitely(const AssignmentParameter& o) const {
  return single->definitely(*o.single);
}

bool bi::AssignmentParameter::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::AssignmentParameter::possibly(const AssignmentParameter& o) const {
  return single->possibly(*o.single);
}
