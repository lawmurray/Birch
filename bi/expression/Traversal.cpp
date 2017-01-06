/**
 * @file
 */
#include "bi/expression/Traversal.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Traversal::Traversal(Expression* left, Expression* right,
    shared_ptr<Location> loc) :
    Expression(loc),
    ExpressionBinary(left, right) {
  //
}

bi::Traversal::~Traversal() {
  //
}

bi::Expression* bi::Traversal::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Traversal::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Traversal::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::Traversal::dispatch(Expression& o) {
  return o.le(*this);
}

bool bi::Traversal::le(Traversal& o) {
  return *left <= *o.left && *right <= *o.right;
}

bool bi::Traversal::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}
