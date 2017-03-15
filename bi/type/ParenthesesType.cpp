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

bool bi::ParenthesesType::dispatchDefinitely(Type& o) {
  return o.definitely(*this) || o.definitely(*single);
}

bool bi::ParenthesesType::definitely(AssignableType& o) {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(BracketsType& o) {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(EmptyType& o) {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(LambdaType& o) {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(List<Type>& o) {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(ModelParameter& o) {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(ModelReference& o) {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(ParenthesesType& o) {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(RandomType& o) {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(VariantType& o) {
  return single->definitely(o);
}

bool bi::ParenthesesType::dispatchPossibly(Type& o) {
  return o.possibly(*this) || o.possibly(*single);
}

bool bi::ParenthesesType::possibly(AssignableType& o) {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(BracketsType& o) {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(EmptyType& o) {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(LambdaType& o) {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(List<Type>& o) {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(ModelParameter& o) {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(ModelReference& o) {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(ParenthesesType& o) {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(RandomType& o) {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(VariantType& o) {
  return single->possibly(o);
}
