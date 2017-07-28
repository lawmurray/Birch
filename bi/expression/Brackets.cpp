/**
 * @file
 */
#include "bi/expression/Brackets.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

bi::Brackets::Brackets(Expression* single, shared_ptr<Location> loc) :
    Expression(loc),
    Unary<Expression>(single) {
  //
}

bi::Brackets::~Brackets() {
  //
}

bi::Expression* bi::Brackets::strip() {
  return single->strip();
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

bool bi::Brackets::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this) || single->dispatchDefinitely(o);
}

bool bi::Brackets::definitely(const Brackets& o) const {
  return single->definitely(*o.single);
}

bool bi::Brackets::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this) || single->dispatchPossibly(o);
}

bool bi::Brackets::possibly(const Brackets& o) const {
  return single->possibly(*o.single);
}
