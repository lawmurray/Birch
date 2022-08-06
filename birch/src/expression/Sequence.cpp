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

void birch::Sequence::accept(Visitor* visitor) const {
  visitor->visit(this);
}
