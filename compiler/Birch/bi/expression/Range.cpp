/**
 * @file
 */
#include "bi/expression/Range.hpp"

#include "bi/visitor/all.hpp"

bi::Range::Range(Expression* left, Expression* right,
    Location* loc) :
    Expression(loc),
    Couple<Expression>(left, right) {
  //
}

bi::Range::~Range() {
  //
}

bi::Expression* bi::Range::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Range::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Range::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
