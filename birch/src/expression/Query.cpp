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

void birch::Query::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
