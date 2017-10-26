/**
 * @file
 */
#include "bi/statement/EmptyStatement.hpp"

#include "bi/visitor/all.hpp"

bi::EmptyStatement::EmptyStatement(Location* loc) :
    Statement(loc) {
  //
}

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
