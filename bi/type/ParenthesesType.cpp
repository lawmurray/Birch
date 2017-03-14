/**
 * @file
 */
#include "bi/type/ParenthesesType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ParenthesesType::ParenthesesType(Type* single, shared_ptr<Location> loc) :
    Type(loc),
    TypeUnary(single) {
  //
}

bi::ParenthesesType::~ParenthesesType() {
  //
}

bool bi::ParenthesesType::isBuiltin() const {
  return single->isBuiltin();
}

bool bi::ParenthesesType::isModel() const {
  return single->isModel();
}

bool bi::ParenthesesType::isRandom() const {
  return single->isRandom();
}

bool bi::ParenthesesType::isLambda() const {
  return single->isLambda();
}

bi::Type* bi::ParenthesesType::strip() {
  return single->strip();
}

bi::Type* bi::ParenthesesType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::ParenthesesType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::ParenthesesType::accept(Visitor* visitor) const {
  visitor->visit(this);
}

bool bi::ParenthesesType::definitely(Type& o) {
  return single->definitely(o);
}

bool bi::ParenthesesType::dispatchDefinitely(Type& o) {
  return o.definitely(*this) || single->dispatchDefinitely(o);
}

bool bi::ParenthesesType::possibly(Type& o) {
  return single->possibly(o);
}

bool bi::ParenthesesType::dispatchPossibly(Type& o) {
  return o.possibly(*this) || single->dispatchPossibly(o);
}
