/**
 * @file
 */
#include "bi/statement/Statement.hpp"

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

bool bi::Statement::isEmpty() const {
  return false;
}
