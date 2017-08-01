/**
 * @file
 */
#include "bi/type/ParenthesesType.hpp"

#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ParenthesesType::ParenthesesType(Type* single, Location* loc,
    const bool assignable) :
    Type(loc, assignable),
    Single<Type>(single) {
  //
}

bi::ParenthesesType::~ParenthesesType() {
  //
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

bool bi::ParenthesesType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::ParenthesesType::definitely(const ParenthesesType& o) const {
  return single->definitely(*o.single);
}

bool bi::ParenthesesType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::ParenthesesType::possibly(const ParenthesesType& o) const {
  return single->possibly(*o.single);
}
