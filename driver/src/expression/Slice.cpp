/**
 * @file
 */
#include "src/expression/Slice.hpp"

#include "src/visitor/all.hpp"

birch::Slice::Slice(Expression* single,
    Expression* brackets, Location* loc) :
    Expression(loc), Single<Expression>(single), Bracketed(brackets) {
  //
}

bool birch::Slice::isAssignable() const {
  return single->isAssignable();
}

birch::Expression* birch::Slice::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Slice::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool birch::Slice::isSlice() const {
  return true;
}
