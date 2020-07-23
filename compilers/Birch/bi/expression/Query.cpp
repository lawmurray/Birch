/**
 * @file
 */
#include "bi/expression/Query.hpp"

#include "bi/visitor/all.hpp"

bi::Query::Query(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

bi::Query::~Query() {
  //
}

bi::Expression* bi::Query::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Query::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Query::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
