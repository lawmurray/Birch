/**
 * @file
 */
#include "bi/statement/EmptyStatement.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

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

bi::possibly bi::EmptyStatement::dispatch(Statement& o) {
  return o.le(*this);
}

bi::possibly bi::EmptyStatement::le(EmptyStatement& o) {
  return definite;
}
