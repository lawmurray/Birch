/**
 * @file
 */
#include "bi/expression/This.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::This::This(shared_ptr<Location> loc) :
    Expression(loc) {
  //
}

bi::This::~This() {
  //
}

bi::Expression* bi::This::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Expression* bi::This::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::This::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::This::dispatch(Expression& o) {
  return o.le(*this);
}

bool bi::This::le(This& o) {
  return *type <= *o.type;
}

bool bi::This::le(VarParameter& o) {
  return *type <= *o.type && o.capture(this);
}
