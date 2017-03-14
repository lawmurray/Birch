/**
 * @file
 */
#include "bi/type/AssignableType.hpp"

#include "bi/visitor/all.hpp"

bi::AssignableType::AssignableType(Type* single, shared_ptr<Location> loc) :
    Type(loc),
    TypeUnary(single) {
  //
}

bi::AssignableType::~AssignableType() {
  //
}

bool bi::AssignableType::isBuiltin() const {
  return single->isBuiltin();
}

bool bi::AssignableType::isModel() const {
  return single->isModel();
}

bool bi::AssignableType::isRandom() const {
  return single->isRandom();
}

bool bi::AssignableType::isLambda() const {
  return single->isLambda();
}

bi::Type* bi::AssignableType::strip() {
  return single->strip();
}

bi::Type* bi::AssignableType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::AssignableType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::AssignableType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::AssignableType::definitely(Type& o) {
  return single->definitely(o);
}

bool bi::AssignableType::dispatchDefinitely(Type& o) {
  return single->dispatchDefinitely(o);
}

bool bi::AssignableType::possibly(Type& o) {
  return single->possibly(o);
}

bool bi::AssignableType::dispatchPossibly(Type& o) {
  return single->dispatchPossibly(o);
}
