/**
 * @file
 */
#include "bi/expression/Parentheses.hpp"

#include "bi/visitor/all.hpp"

bi::Parentheses::Parentheses(Expression* single,
    Location* loc) :
    Expression(loc),
    Single<Expression>(single) {
  //
}

bi::Parentheses::~Parentheses() {
  //
}

bi::Expression* bi::Parentheses::strip() {
  return single->strip();
}

bi::Iterator<bi::Expression> bi::Parentheses::begin() const {
  return single->begin();
}

bi::Iterator<bi::Expression> bi::Parentheses::end() const {
  return single->end();
}

bi::Expression* bi::Parentheses::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Parentheses::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Parentheses::accept(Visitor* visitor) const {
  visitor->visit(this);
}
