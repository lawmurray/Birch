/**
 * @file
 */
#include "bi/expression/Span.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Span::Span(Expression* single, shared_ptr<Location> loc) :
    Expression(loc),
    Unary<Expression>(single) {
  //
}

bi::Span::~Span() {
  //
}

bi::Expression* bi::Span::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Span::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Span::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::Span::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Span::definitely(const Span& o) const {
  return single->definitely(*o.single);
}

bool bi::Span::definitely(const VarParameter& o) const {
  /* transparent to capture */
  return single->definitely(o);
}

bool bi::Span::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Span::possibly(const Span& o) const {
  return single->possibly(*o.single);
}

bool bi::Span::possibly(const VarParameter& o) const {
  /* transparent to capture */
  return single->possibly(o);
}
