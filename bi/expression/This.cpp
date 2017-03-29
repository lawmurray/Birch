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

bool bi::This::dispatchDefinitely(const Expression& o) const {
  return o.definitely(*this);
}

bool bi::This::definitely(const This& o) const {
  return type->definitely(*o.type);
}

bool bi::This::definitely(const VarParameter& o) const {
  return type->definitely(*o.type);
}

bool bi::This::dispatchPossibly(const Expression& o) const {
  return o.possibly(*this);
}

bool bi::This::possibly(const This& o) const {
  return type->possibly(*o.type);
}

bool bi::This::possibly(const VarParameter& o) const {
  return type->possibly(*o.type);
}
