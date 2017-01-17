/**
 * @file
 */
#include "bi/expression/BracesExpression.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::BracesExpression::BracesExpression(Statement* single,
    shared_ptr<Location> loc) :
    Expression(loc),
    StatementUnary(single) {
  //
}

bi::BracesExpression::~BracesExpression() {
  //
}

bi::Expression* bi::BracesExpression::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::BracesExpression::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::BracesExpression::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bi::possibly bi::BracesExpression::dispatch(Expression& o) {
  return o.le(*this);
}

bi::possibly bi::BracesExpression::le(BracesExpression& o) {
  return *single <= *o.single && *type <= *o.type;
}
