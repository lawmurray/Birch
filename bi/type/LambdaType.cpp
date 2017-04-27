/**
 * @file
 */
#include "bi/type/LambdaType.hpp"

#include "bi/visitor/all.hpp"

bi::LambdaType::LambdaType(Type* parens, Type* type,
    shared_ptr<Location> loc, const bool assignable) :
    Type(loc, assignable),
    parens(parens),
    type(type) {
  //
}

bi::LambdaType::~LambdaType() {
  //
}

bi::Type* bi::LambdaType::accept(Cloner* visitor) const {
  return visitor->clone(this);
}

bi::Type* bi::LambdaType::accept(Modifier* visitor) {
  return visitor->modify(this);
}

void bi::LambdaType::accept(Visitor* visitor) const {
  return visitor->visit(this);
}

bool bi::LambdaType::isLambda() const {
  return true;
}

bool bi::LambdaType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this);
}

bool bi::LambdaType::definitely(const LambdaType& o) const {
  return parens->definitely(*o.parens) && type->definitely(*o.type);
}

bool bi::LambdaType::definitely(const ParenthesesType& o) const {
  return definitely(*o.single);
}

bool bi::LambdaType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this);
}

bool bi::LambdaType::possibly(const LambdaType& o) const {
  return parens->possibly(*o.parens) && type->possibly(*o.type);
}

bool bi::LambdaType::possibly(const ParenthesesType& o) const {
  return possibly(*o.single);
}
