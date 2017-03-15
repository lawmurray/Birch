/**
 * @file
 */
#include "bi/type/LambdaType.hpp"

#include "bi/visitor/all.hpp"

bi::LambdaType::LambdaType(Type* result, shared_ptr<Location> loc) :
    Type(loc),
    result(result) {
  /* pre-conditions */
  assert(result);
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
  return o.definitely(*this) || o.definitely(*result);
}

bool bi::LambdaType::definitely(AssignableType& o) {
  return result->definitely(o);
}

bool bi::LambdaType::definitely(BracketsType& o) {
  return result->definitely(o);
}

bool bi::LambdaType::definitely(EmptyType& o) {
  return result->definitely(o);
}

bool bi::LambdaType::definitely(LambdaType& o) {
  return result->definitely(o);
}

bool bi::LambdaType::definitely(List<Type>& o) {
  return result->definitely(o);
}

bool bi::LambdaType::definitely(ModelParameter& o) {
  return result->definitely(o);
}

bool bi::LambdaType::definitely(ModelReference& o) {
  return result->definitely(o);
}

bool bi::LambdaType::definitely(ParenthesesType& o) {
  return result->definitely(o);
}

bool bi::LambdaType::definitely(RandomType& o) {
  return result->definitely(o);
}

bool bi::LambdaType::definitely(VariantType& o) {
  return result->definitely(o);
}

bool bi::LambdaType::dispatchPossibly(Type& o) {
  return o.possibly(*this) || o.possibly(*result);
}

bool bi::LambdaType::possibly(AssignableType& o) {
  return result->possibly(o);
}

bool bi::LambdaType::possibly(BracketsType& o) {
  return result->possibly(o);
}

bool bi::LambdaType::possibly(EmptyType& o) {
  return result->possibly(o);
}

bool bi::LambdaType::possibly(LambdaType& o) {
  return result->possibly(o);
}

bool bi::LambdaType::possibly(List<Type>& o) {
  return result->possibly(o);
}

bool bi::LambdaType::possibly(ModelParameter& o) {
  return result->possibly(o);
}

bool bi::LambdaType::possibly(ModelReference& o) {
  return result->possibly(o);
}

bool bi::LambdaType::possibly(ParenthesesType& o) {
  return result->possibly(o);
}

bool bi::LambdaType::possibly(RandomType& o) {
  return result->possibly(o);
}

bool bi::LambdaType::possibly(VariantType& o) {
  return result->possibly(o);
}
