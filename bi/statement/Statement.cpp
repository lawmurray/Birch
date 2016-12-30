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

bi::Statement::operator bool() const {
  return true;
}

bool bi::Statement::operator<(Statement& o) {
  return *this <= o && *this != o;
}

bool bi::Statement::operator>(Statement& o) {
  return o <= *this && o != *this;
}

bool bi::Statement::operator>=(Statement& o) {
  return o <= *this;
}

bool bi::Statement::operator!=(Statement& o) {
  return !(*this == o);
}
