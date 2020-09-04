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

birch::Span::~Span() {
  //
}

birch::Expression* birch::Span::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Expression* birch::Span::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Span::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
