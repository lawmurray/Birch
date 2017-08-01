/**
 * @file
 */
#include "bi/statement/Yield.hpp"

#include "bi/visitor/all.hpp"

bi::Yield::Yield(Expression* single,
    Location* loc) :
    Statement(loc),
    Unary<Expression>(single) {
  //
}

bi::Yield::~Yield() {
  //
}

bi::Statement* bi::Yield::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Statement* bi::Yield::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Yield::accept(Visitor* visitor) const {
  visitor->visit(this);
}
