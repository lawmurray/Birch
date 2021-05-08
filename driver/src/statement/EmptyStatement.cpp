/**
 * @file
 */
#include "src/statement/EmptyStatement.hpp"

#include "src/visitor/all.hpp"

birch::EmptyStatement::EmptyStatement(Location* loc) :
    Statement(loc) {
  //
}

void birch::EmptyStatement::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool birch::EmptyStatement::isEmpty() const {
  return true;
}
