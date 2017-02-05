/**
 * @file
 */
#include "bi/expression/Index.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Index::Index(Expression* single, shared_ptr<Location> loc) :
    Expression(loc),
    ExpressionUnary(single) {
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

bool bi::Index::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::Index::definitely(Index& o) {
  return single->definitely(*o.single);
}

bool bi::Index::definitely(VarParameter& o) {
  /* transparent to capture */
  return single->definitely(o);
}

bool bi::Index::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::Index::possibly(Index& o) {
  return single->possibly(*o.single);
}

bool bi::Index::possibly(VarParameter& o) {
  /* transparent to capture */
  return single->possibly(o);
}
