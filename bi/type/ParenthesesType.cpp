/**
 * @file
 */
#include "bi/type/ParenthesesType.hpp"

#include "bi/type/ModelParameter.hpp"
#include "bi/visitor/all.hpp"

#include <typeinfo>

bi::ParenthesesType::ParenthesesType(Type* single, shared_ptr<Location> loc,
    const bool assignable, const bool polymorphic) :
    Type(loc, assignable, polymorphic),
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

bool bi::ParenthesesType::isDelay() const {
  return single->isDelay();
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

bool bi::ParenthesesType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this) || o.definitely(*single);
}

bool bi::ParenthesesType::definitely(const AssignableType& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(const BracketsType& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(const EmptyType& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(const LambdaType& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(const List<Type>& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(const ModelParameter& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(const ModelReference& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(const ParenthesesType& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(const DelayType& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesType::definitely(const VariantType& o) const {
  return single->definitely(o);
}

bool bi::ParenthesesType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this) || o.possibly(*single);
}

bool bi::ParenthesesType::possibly(const AssignableType& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(const BracketsType& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(const EmptyType& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(const LambdaType& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(const List<Type>& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(const ModelParameter& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(const ModelReference& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(const ParenthesesType& o) const {
  return single->possibly(*o.single);
}

bool bi::ParenthesesType::possibly(const DelayType& o) const {
  return single->possibly(o);
}

bool bi::ParenthesesType::possibly(const VariantType& o) const {
  return single->possibly(o);
}
