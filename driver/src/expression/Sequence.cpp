/**
 * @file
 */
#include "src/expression/Sequence.hpp"

#include "src/visitor/all.hpp"

birch::Sequence::Sequence(Expression* single,
    Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

birch::Sequence::~Sequence() {
  //
}

birch::Expression* birch::Sequence::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

birch::Expression* birch::Sequence::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void birch::Sequence::accept(Visitor* visitor) const {
  visitor->visit(this);
}
