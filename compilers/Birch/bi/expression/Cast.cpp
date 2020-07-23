/**
 * @file
 */
#include "bi/expression/Cast.hpp"

#include "bi/visitor/all.hpp"

bi::Cast::Cast(Type* returnType, Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single),
    ReturnTyped(returnType) {
  //
}

bi::Cast::~Cast() {
  //
}

bi::Expression* bi::Cast::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Cast::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Cast::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
