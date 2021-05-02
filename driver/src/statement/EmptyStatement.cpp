/**
 * @file
 */
#include "src/statement/EmptyStatement.hpp"

#include "src/visitor/all.hpp"

birch::EmptyStatement::EmptyStatement(Location* loc) :
    Statement(loc) {
  //
}

birch::Statement* birch::EmptyStatement::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::EmptyStatement::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool birch::EmptyStatement::isEmpty() const {
  return true;
}
