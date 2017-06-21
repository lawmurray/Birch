/**
 * @file
 */
#include "bi/statement/EmptyStatement.hpp"

#include "bi/visitor/all.hpp"

bi::EmptyStatement::~EmptyStatement() {
  //
}

bi::Statement* bi::EmptyStatement::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::EmptyStatement::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::EmptyStatement::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::EmptyStatement::isEmpty() const {
  return true;
}

bool bi::EmptyStatement::dispatchDefinitely(const Statement& o) const {
  return o.definitely(*this);
}

bool bi::EmptyStatement::definitely(const EmptyStatement& o) const {
  return true;
}

bool bi::EmptyStatement::dispatchPossibly(const Statement& o) const {
  return o.possibly(*this);
}

bool bi::EmptyStatement::possibly(const EmptyStatement& o) const {
  return true;
}
