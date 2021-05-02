/**
 * @file
 */
#include "src/expression/Query.hpp"

#include "src/visitor/all.hpp"

birch::Query::Query(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

birch::Expression* birch::Query::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Query::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
