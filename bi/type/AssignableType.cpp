/**
 * @file
 */
#include "bi/type/AssignableType.hpp"

#include "bi/visitor/all.hpp"

bi::AssignableType::AssignableType(Type* single, shared_ptr<Location> loc,
    const bool assignable, const bool polymorphic) :
    Type(loc, assignable, polymorphic),
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

bool bi::AssignableType::isDelay() const {
  return single->isDelay();
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

bool bi::AssignableType::dispatchDefinitely(const Type& o) const {
  return o.definitely(*this) || o.definitely(*single);
}

bool bi::AssignableType::definitely(const AssignableType& o) const {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(const BracketsType& o) const {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(const EmptyType& o) const {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(const LambdaType& o) const {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(const List<Type>& o) const {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(const ModelParameter& o) const {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(const ModelReference& o) const {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(const ParenthesesType& o) const {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(const DelayType& o) const {
  return single->definitely(o);
}

bool bi::AssignableType::definitely(const VariantType& o) const {
  return single->definitely(o);
}

bool bi::AssignableType::dispatchPossibly(const Type& o) const {
  return o.possibly(*this) || o.possibly(*single);
}

bool bi::AssignableType::possibly(const AssignableType& o) const {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(const BracketsType& o) const {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(const EmptyType& o) const {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(const LambdaType& o) const {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(const List<Type>& o) const {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(const ModelParameter& o) const {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(const ModelReference& o) const {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(const ParenthesesType& o) const {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(const DelayType& o) const {
  return single->possibly(o);
}

bool bi::AssignableType::possibly(const VariantType& o) const {
  return single->possibly(o);
}
