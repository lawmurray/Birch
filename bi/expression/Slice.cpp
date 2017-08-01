/**
 * @file
 */
#include "bi/expression/Slice.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::Slice::Slice(Expression* single,
    Expression* brackets, Location* loc) :
    Expression(loc), Unary<Expression>(single), Bracketed(brackets) {
  //
}

bi::Slice::~Slice() {
  //
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
