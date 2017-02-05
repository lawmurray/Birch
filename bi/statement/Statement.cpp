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

bool bi::Statement::definitely(Statement& o) {
  return o.dispatchDefinitely(*this);
}

bool bi::Statement::definitely(Conditional& o) {
  return false;
}

bool bi::Statement::definitely(Declaration<VarParameter>& o) {
  return false;
}

bool bi::Statement::definitely(Declaration<FuncParameter>& o) {
  return false;
}

bool bi::Statement::definitely(Declaration<ProgParameter>& o) {
  return false;
}

bool bi::Statement::definitely(Declaration<ModelParameter>& o) {
  return false;
}

bool bi::Statement::definitely(EmptyStatement& o) {
  return false;
}

bool bi::Statement::definitely(ExpressionStatement& o) {
  return false;
}

bool bi::Statement::definitely(Import& o) {
  return false;
}

bool bi::Statement::definitely(List<Statement>& o) {
  return false;
}

bool bi::Statement::definitely(Loop& o) {
  return false;
}

bool bi::Statement::definitely(Raw& o) {
  return false;
}

bool bi::Statement::possibly(Statement& o) {
  return o.dispatchPossibly(*this);
}

bool bi::Statement::possibly(Conditional& o) {
  return false;
}

bool bi::Statement::possibly(Declaration<VarParameter>& o) {
  return false;
}

bool bi::Statement::possibly(Declaration<FuncParameter>& o) {
  return false;
}

bool bi::Statement::possibly(Declaration<ProgParameter>& o) {
  return false;
}

bool bi::Statement::possibly(Declaration<ModelParameter>& o) {
  return false;
}

bool bi::Statement::possibly(EmptyStatement& o) {
  return false;
}

bool bi::Statement::possibly(ExpressionStatement& o) {
  return false;
}

bool bi::Statement::possibly(Import& o) {
  return false;
}

bool bi::Statement::possibly(List<Statement>& o) {
  return false;
}

bool bi::Statement::possibly(Loop& o) {
  return false;
}

bool bi::Statement::possibly(Raw& o) {
  return false;
}
