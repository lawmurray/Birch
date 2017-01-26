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

bi::possibly bi::Index::dispatch(Expression& o) {
  return o.le(*this);
}

bi::possibly bi::Index::le(Index& o) {
  return *single <= *o.single;
}

bi::possibly bi::Index::le(VarParameter& o) {
  /* transparent to capture */
  return *single <= o;
}
