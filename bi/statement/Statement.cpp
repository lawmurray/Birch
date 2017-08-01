/**
 * @file
 */
#include "bi/statement/Statement.hpp"

bi::Statement::Statement(shared_ptr<Location> loc) :
    Located(loc) {
  //
}

bi::Statement::~Statement() {
  //
}

bool bi::Statement::isEmpty() const {
  return false;
}
