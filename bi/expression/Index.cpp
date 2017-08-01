/**
 * @file
 */
#include "bi/expression/Index.hpp"

#include "bi/visitor/all.hpp"

bi::Index::Index(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

bi::Index::~Index() {
  //
}

bi::Expression* bi::Index::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Index::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Index::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
