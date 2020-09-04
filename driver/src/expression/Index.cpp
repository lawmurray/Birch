/**
 * @file
 */
#include "src/expression/Index.hpp"

#include "src/visitor/all.hpp"

birch::Index::Index(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

birch::Index::~Index() {
  //
}

birch::Expression* birch::Index::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Expression* birch::Index::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Index::accept(Visitor* visitor) const {
  return visitor->visit(this);
}
