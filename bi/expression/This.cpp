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

bool bi::This::dispatchDefinitely(Expression& o) {
  return o.definitely(*this);
}

bool bi::This::definitely(This& o) {
  return type->definitely(*o.type);
}

bool bi::This::definitely(VarParameter& o) {
  return type->definitely(*o.type);
}

bool bi::This::dispatchPossibly(Expression& o) {
  return o.possibly(*this);
}

bool bi::This::possibly(This& o) {
  return type->possibly(*o.type);
}

bool bi::This::possibly(VarParameter& o) {
  return type->possibly(*o.type);
}
