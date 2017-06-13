/**
 * @file
 */
#include "bi/expression/Index.hpp"

#include "bi/expression/Parameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Index::Index(Expression* single, shared_ptr<Location> loc) :
    Expression(loc),
    Unary<Expression>(single) {
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

bool bi::Index::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::Index::definitely(const Index& o) const {
  return single->definitely(*o.single);
}

bool bi::Index::definitely(const Parameter& o) const {
  /* transparent to capture */
  return single->definitely(o);
}

bool bi::Index::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::Index::possibly(const Index& o) const {
  return single->possibly(*o.single);
}

bool bi::Index::possibly(const Parameter& o) const {
  /* transparent to capture */
  return single->possibly(o);
}
