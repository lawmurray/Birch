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

bool bi::Statement::operator<=(Statement& o) {
  return o.dispatch(*this);
}

bool bi::Statement::operator==(Statement& o) {
  return *this <= o && o <= *this;
}

bool bi::Statement::le(Conditional& o) {
  return false;
}

bool bi::Statement::le(Declaration<VarParameter>& o) {
  return false;
}

bool bi::Statement::le(Declaration<FuncParameter>& o) {
  return false;
}

bool bi::Statement::le(Declaration<ProgParameter>& o) {
  return false;
}

bool bi::Statement::le(Declaration<ModelParameter>& o) {
  return false;
}

bool bi::Statement::le(EmptyStatement& o) {
  return false;
}

bool bi::Statement::le(ExpressionStatement& o) {
  return false;
}

bool bi::Statement::le(Import& o) {
  return false;
}

bool bi::Statement::le(List<Statement>& o) {
  return false;
}

bool bi::Statement::le(Loop& o) {
  return false;
}

bool bi::Statement::le(Raw& o) {
  return false;
}
