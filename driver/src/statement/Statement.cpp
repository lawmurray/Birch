/**
 * @file
 */
#include "src/statement/Statement.hpp"

#include "src/statement/StatementIterator.hpp"
#include "src/visitor/all.hpp"

birch::Statement::Statement(Location* loc) :
    Located(loc) {
  //
}

birch::Statement::~Statement() {
  //
}

birch::Statement* birch::Statement::strip() {
  return this;
}

int birch::Statement::count() const {
  return 1;
}

bool birch::Statement::isEmpty() const {
  return false;
}

birch::StatementIterator birch::Statement::begin() const {
  if (isEmpty()) {
    return end();
  } else {
    return StatementIterator(this);
  }
}

birch::StatementIterator birch::Statement::end() const {
  return StatementIterator(nullptr);
}
