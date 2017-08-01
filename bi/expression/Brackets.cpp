/**
 * @file
 */
#include "bi/expression/Brackets.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::Brackets::Brackets(Expression* single, Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

bi::Brackets::~Brackets() {
  //
}

bi::Iterator<bi::Expression> bi::Brackets::begin() const {
  return single->begin();
}

bi::Iterator<bi::Expression> bi::Brackets::end() const {
  return single->end();
}

bi::Expression* bi::Brackets::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Brackets::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Brackets::accept(Visitor* visitor) const {
  visitor->visit(this);
}
