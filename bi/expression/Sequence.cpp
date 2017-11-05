/**
 * @file
 */
#include "bi/expression/Sequence.hpp"

#include "bi/visitor/all.hpp"

bi::Sequence::Sequence(Expression* single,
    Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

bi::Sequence::~Sequence() {
  //
}

bi::Expression* bi::Sequence::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Sequence::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Sequence::accept(Visitor* visitor) const {
  visitor->visit(this);
}
