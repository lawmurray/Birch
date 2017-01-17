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

bi::possibly bi::Statement::operator<=(Statement& o) {
  return o.dispatch(*this);
}

bi::possibly bi::Statement::operator==(Statement& o) {
  return *this <= o && o <= *this;
}

bi::possibly bi::Statement::le(Conditional& o) {
  return untrue;
}

bi::possibly bi::Statement::le(Declaration<VarParameter>& o) {
  return untrue;
}

bi::possibly bi::Statement::le(Declaration<FuncParameter>& o) {
  return untrue;
}

bi::possibly bi::Statement::le(Declaration<ProgParameter>& o) {
  return untrue;
}

bi::possibly bi::Statement::le(Declaration<ModelParameter>& o) {
  return untrue;
}

bi::possibly bi::Statement::le(EmptyStatement& o) {
  return untrue;
}

bi::possibly bi::Statement::le(ExpressionStatement& o) {
  return untrue;
}

bi::possibly bi::Statement::le(Import& o) {
  return untrue;
}

bi::possibly bi::Statement::le(List<Statement>& o) {
  return untrue;
}

bi::possibly bi::Statement::le(Loop& o) {
  return untrue;
}

bi::possibly bi::Statement::le(Raw& o) {
  return untrue;
}
