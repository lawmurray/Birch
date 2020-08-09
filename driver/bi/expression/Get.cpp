/**
 * @file
 */
#include "bi/expression/Get.hpp"

#include "bi/visitor/all.hpp"

bi::Get::Get(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

bi::Get::~Get() {
  //
}

bool bi::Get::isAssignable() const {
  return single->isAssignable();
}

bi::Expression* bi::Get::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Get::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Get::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
