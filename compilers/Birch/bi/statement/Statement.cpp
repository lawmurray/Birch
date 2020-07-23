/**
 * @file
 */
#include "bi/statement/Statement.hpp"

#include "bi/statement/StatementIterator.hpp"
#include "bi/visitor/all.hpp"

bi::Statement::Statement(Location* loc) :
    Located(loc) {
  //
}

bi::Statement::~Statement() {
  //
}

bi::Statement* bi::Statement::strip() {
  return this;
}

int bi::Statement::count() const {
  return 1;
}

bool bi::Statement::isEmpty() const {
  return false;
}

bool bi::Statement::isDeclaration() const {
  return false;
}

bi::StatementIterator bi::Statement::begin() const {
  if (isEmpty()) {
    return end();
  } else {
    return StatementIterator(this);
  }
}

bi::StatementIterator bi::Statement::end() const {
  return StatementIterator(nullptr);
}
