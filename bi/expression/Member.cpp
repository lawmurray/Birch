/**
 * @file
 */
#include "bi/expression/Member.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Member::Member(Expression* left, Expression* right,
    Location* loc) :
    Expression(loc),
    Couple<Expression>(left, right) {
  //
}

bi::Member::~Member() {
  //
}

bi::Expression* bi::Member::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Member::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Member::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
