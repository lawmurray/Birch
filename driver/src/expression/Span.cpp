/**
 * @file
 */
#include "src/expression/Span.hpp"

#include "src/visitor/all.hpp"

birch::Span::Span(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

void birch::Span::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
