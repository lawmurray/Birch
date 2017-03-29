/**
 * @file
 */
#include "bi/type/LambdaType.hpp"

#include "bi/visitor/all.hpp"

bi::LambdaType::LambdaType(Type* parens, Type* result,
    shared_ptr<Location> loc, const bool assignable) :
    Type(loc, assignable),
    parens(parens),
    result(result) {
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

bool bi::LambdaType::dispatchDefinitely(Type& o) {
  return o.definitely(*this);
}

bool bi::LambdaType::definitely(LambdaType& o) {
  return parens->definitely(*o.parens) && result->definitely(*o.result);
}

bool bi::LambdaType::dispatchPossibly(Type& o) {
  return o.possibly(*this);
}

bool bi::LambdaType::possibly(LambdaType& o) {
  return parens->possibly(*o.parens) && result->possibly(*o.result);
}
