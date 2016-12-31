/**
 * @file
 */
#include "bi/statement/EmptyStatement.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::EmptyStatement::~EmptyStatement() {
  //
}

bi::Statement* bi::EmptyStatement::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::EmptyStatement::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::EmptyStatement::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bi::EmptyStatement::operator bool() const {
  return false;
}

bool bi::EmptyStatement::operator<=(Statement& o) {
  try {
    EmptyStatement& o1 = dynamic_cast<EmptyStatement&>(o);
    return true;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::EmptyStatement::operator==(const Statement& o) const {
  try {
    const EmptyStatement& o1 = dynamic_cast<const EmptyStatement&>(o);
    return true;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}
