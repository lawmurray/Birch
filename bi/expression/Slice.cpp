/**
 * @file
 */
#include "bi/expression/Slice.hpp"

#include "bi/visitor/all.hpp"

bi::Slice::Slice(Expression* single,
    Expression* brackets, Location* loc) :
    Expression(loc), Single<Expression>(single), Bracketed(brackets) {
  //
}

bi::Slice::~Slice() {
  //
}

bool bi::Slice::isAssignable() const {
  return single->isAssignable();
}

bi::Expression* bi::Slice::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Slice::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Slice::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::Slice::isSlice() const {
  return true;
}
