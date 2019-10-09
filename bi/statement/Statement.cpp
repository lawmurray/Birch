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

bool bi::Statement::isValue() const {
  IsValue visitor;
  accept(&visitor);
  return visitor.result;
}

bool bi::Statement::isEmpty() const {
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

int bi::Statement::count() const {
  int count = 0;
  auto iter = begin();
  while (iter != end()) {
    ++count;
    ++iter;
  }
  return count;
}
