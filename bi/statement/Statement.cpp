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

bool bi::Statement::definitely(const Statement& o) const {
  return o.dispatchDefinitely(*this);
}

bool bi::Statement::definitely(const Conditional& o) const {
  return false;
}

bool bi::Statement::definitely(const Declaration<VarParameter>& o) const {
  return false;
}

bool bi::Statement::definitely(const Declaration<FuncParameter>& o) const {
  return false;
}

bool bi::Statement::definitely(const Declaration<ProgParameter>& o) const {
  return false;
}

bool bi::Statement::definitely(const Declaration<TypeParameter>& o) const {
  return false;
}

bool bi::Statement::definitely(const EmptyStatement& o) const {
  return false;
}

bool bi::Statement::definitely(const ExpressionStatement& o) const {
  return false;
}

bool bi::Statement::definitely(const Import& o) const {
  return false;
}

bool bi::Statement::definitely(const List<Statement>& o) const {
  return false;
}

bool bi::Statement::definitely(const Loop& o) const {
  return false;
}

bool bi::Statement::definitely(const Raw& o) const {
  return false;
}

bool bi::Statement::possibly(const Statement& o) const {
  return o.dispatchPossibly(*this);
}

bool bi::Statement::possibly(const Conditional& o) const {
  return false;
}

bool bi::Statement::possibly(const Declaration<VarParameter>& o) const {
  return false;
}

bool bi::Statement::possibly(const Declaration<FuncParameter>& o) const {
  return false;
}

bool bi::Statement::possibly(const Declaration<ProgParameter>& o) const {
  return false;
}

bool bi::Statement::possibly(const Declaration<TypeParameter>& o) const {
  return false;
}

bool bi::Statement::possibly(const EmptyStatement& o) const {
  return false;
}

bool bi::Statement::possibly(const ExpressionStatement& o) const {
  return false;
}

bool bi::Statement::possibly(const Import& o) const {
  return false;
}

bool bi::Statement::possibly(const List<Statement>& o) const {
  return false;
}

bool bi::Statement::possibly(const Loop& o) const {
  return false;
}

bool bi::Statement::possibly(const Raw& o) const {
  return false;
}
