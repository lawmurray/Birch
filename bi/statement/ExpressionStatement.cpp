/**
 * @file
 */
#include "bi/statement/ExpressionStatement.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ExpressionStatement::ExpressionStatement(Expression* expr,
    shared_ptr<Location> loc) :
    Statement(loc), expr(expr) {
  //
}

bi::ExpressionStatement::~ExpressionStatement() {
  //
}

bi::Statement* bi::ExpressionStatement::acceptClone(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::ExpressionStatement::acceptModify(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ExpressionStatement::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ExpressionStatement::operator<=(Statement& o) {
  try {
    ExpressionStatement& o1 = dynamic_cast<ExpressionStatement&>(o);
    return *expr <= *o1.expr;
  } catch (std::bad_cast e) {
    //
  }
  return false;
}

bool bi::ExpressionStatement::operator==(const Statement& o) const {
  try {
    const ExpressionStatement& o1 =
        dynamic_cast<const ExpressionStatement&>(o);
    return *expr == *o1.expr;
  } catch (std::bad_cast e) {
    return false;
  }
}
