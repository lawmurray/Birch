/**
 * @file
 */
#include "bi/expression/Range.hpp"

#include "bi/expression/VarParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::Range::Range(Expression* left, Expression* right,
    shared_ptr<Location> loc) :
    Expression(loc),
    ExpressionBinary(left, right) {
  //
}

bi::Range::~Range() {
  //
}

bi::Expression* bi::Range::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::Range::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::Range::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bi::possibly bi::Range::dispatch(Expression& o) {
  return o.le(*this);
}

bi::possibly bi::Range::le(Range& o) {
  return *left <= *o.left && *right <= *o.right;
}
